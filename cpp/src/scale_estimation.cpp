/**
 * Scale estimation: variance mapping, scale matching, per-segment metric depth (Eq. 12)
 */
#include "ptc_depth/scale_estimation.hpp"
#include "ptc_depth/utils.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ptc_depth {

// ============================================================================
// LabelIndex: counting-sort pixel index by segment label
// ============================================================================

LabelIndex build_label_index(const cv::Mat& labels, int num_labels) {
    LabelIndex idx;
    if (labels.empty()) return idx;

    const int H = labels.rows, W = labels.cols;
    if (num_labels <= 0) {
        int maxL = 0;
        for (int r = 0; r < H; ++r) {
            const int32_t* row = labels.ptr<int32_t>(r);
            for (int c = 0; c < W; ++c)
                maxL = std::max(maxL, row[c]);
        }
        num_labels = maxL + 1;
    }

    idx.num_labels = num_labels;
    idx.offsets.resize(num_labels + 1, 0);

    for (int r = 0; r < H; ++r) {
        const int32_t* row = labels.ptr<int32_t>(r);
        for (int c = 0; c < W; ++c) {
            int L = row[c];
            if (L >= 0 && L < num_labels) idx.offsets[L + 1]++;
        }
    }
    for (int L = 1; L <= num_labels; ++L)
        idx.offsets[L] += idx.offsets[L - 1];

    idx.flat_pixels.resize(idx.offsets[num_labels]);
    std::vector<int> wp(idx.offsets.begin(), idx.offsets.begin() + num_labels);
    for (int r = 0; r < H; ++r) {
        const int32_t* row = labels.ptr<int32_t>(r);
        for (int c = 0; c < W; ++c) {
            int L = row[c];
            if (L >= 0 && L < num_labels)
                idx.flat_pixels[wp[L]++] = r * W + c;
        }
    }
    return idx;
}

// ============================================================================
// rho_to_variance
// ============================================================================

cv::Mat rho_to_variance(
    const cv::Mat& rho, const CameraIntrinsics& cam,
    float baseline, float b_ref_auto,
    const FusionConfig& config
) {
    if (rho.empty()) return cv::Mat();

    const int H = rho.rows, W = rho.cols;
    const float f_eff = std::sqrt(std::max(cam.fx, kEpsDenomF) * std::max(cam.fy, kEpsDenomF));
    float tau0 = (config.tau0_deg > 0) ? config.tau0_deg * static_cast<float>(M_PI) / 180.0f : 1.75e-3f;
    float B = (std::isfinite(baseline) && baseline > 0) ? baseline : b_ref_auto;
    float bref = (std::isfinite(b_ref_auto) && b_ref_auto > 0) ? b_ref_auto : 1.0f;
    tau0 *= (B / bref);
    const float scale = config.sigma2_at_tau / (std::max(tau0, kEpsDenomF) * std::max(tau0, kEpsDenomF) * f_eff * f_eff);

    cv::Mat variance;
    cv::max(rho, 0.0f, variance);
    variance *= scale;
    cv::min(variance, config.max_var, variance);
    cv::max(variance, config.min_var, variance);
    // NaN residuals → max variance (invalid triangulation)
    cv::Mat nan_mask;
    cv::compare(rho, rho, nan_mask, cv::CMP_NE);  // NaN != NaN → true
    variance.setTo(config.max_var, nan_mask);
    return variance;
}

// Forward declaration
static double robust_median(std::vector<double>& ratios, double trim_k);

// ============================================================================
// BaselineAutoState
// ============================================================================

BaselineAutoState::BaselineAutoState(float ema_beta, int hist_len)
    : b_ema_(0.0f), has_ema_(false), beta_(ema_beta), hist_len_(hist_len) {}

void BaselineAutoState::update(float b) {
    if (!std::isfinite(b)) return;
    if (!has_ema_) { b_ema_ = b; has_ema_ = true; }
    else { b_ema_ = beta_ * b_ema_ + (1.0f - beta_) * b; }
    hist_.push_back(b);
    if (static_cast<int>(hist_.size()) > hist_len_) hist_.pop_front();
}

bool BaselineAutoState::check(float baseline, float min_baseline) {
    update(baseline);
    return !std::isfinite(baseline) || baseline < min_baseline;
}

float BaselineAutoState::b_ref() const {
    if (has_ema_) return b_ema_;
    if (!hist_.empty()) {
        std::vector<double> h(hist_.begin(), hist_.end());
        return static_cast<float>(compute_median(h));
    }
    return 0.1f;
}


// ============================================================================
// Robust median with MAD trimming
// trim_k = 3.5 ≈ 99.95% retention for Gaussian (standard robust statistics)
// ============================================================================

static double robust_median(std::vector<double>& ratios, double trim_k) {
    if (ratios.empty()) return 1.0;
    double med = compute_median(ratios);
    double mad = compute_mad(ratios);
    if (mad <= 0) return med;

    std::vector<double> inliers;
    inliers.reserve(ratios.size());
    for (double r : ratios)
        if (std::abs(r - med) <= trim_k * mad) inliers.push_back(r);

    return inliers.empty() ? med : compute_median(inliers);
}

// ============================================================================
// solve_metric_from_rel
// ============================================================================

static MetricScaleResult solve_metric_core(
    const cv::Mat& d_rel, const cv::Mat& z_obs,
    const cv::Mat& mask, const LabelIndex& label_index,
    const cv::Mat& v_px, const MetricScaleConfig& config
) {
    const int H = d_rel.rows, W = d_rel.cols, total = H * W;

    MetricScaleResult result;
    result.z_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    result.V_out = cv::Mat(H, W, CV_32F, cv::Scalar(100.0f));

    const bool has_mask = !mask.empty(), has_vpx = !v_px.empty();
    const float* rd = d_rel.ptr<float>(0);
    const float* zo = z_obs.ptr<float>(0);
    const uint8_t* mk = has_mask ? mask.ptr<uint8_t>(0) : nullptr;
    const float* vp = has_vpx ? v_px.ptr<float>(0) : nullptr;

    // Collect ratios + RANSAC samples
    std::vector<double> ratios;
    std::vector<float> fit_x, fit_y;
    std::vector<int> fit_idx;
    ratios.reserve(total / 2);
    fit_x.reserve(total / 4);

    for (int i = 0; i < total; ++i) {
        if (has_mask && mk[i] == 0) continue;
        if (!std::isfinite(rd[i]) || !std::isfinite(zo[i]) || rd[i] <= 0 || zo[i] <= 0) continue;

        ratios.push_back(static_cast<double>(zo[i]) / static_cast<double>(rd[i]));

        if ((i & 1) == 0 && rd[i] > 1e-6f && zo[i] > 0.1f && zo[i] < 80.0f) {
            fit_x.push_back(1.0f / rd[i]);
            fit_y.push_back(1.0f / zo[i]);
            fit_idx.push_back(i);
        }
    }

    if (ratios.empty()) {
        // Fallback: fill with whatever is available
        float* out = result.z_out.ptr<float>(0);
        for (int i = 0; i < total; ++i)
            out[i] = (std::isfinite(zo[i]) && zo[i] > 0) ? zo[i]
                   : (std::isfinite(rd[i]) && rd[i] > 0) ? rd[i] : 1.0f;
        return result;
    }

    // === Global scale via RANSAC + MAD fallback ===
    std::vector<double> inlier_ratios;
    std::vector<int> inlier_pixel_idx;  // pixel indices of scale inliers
    if (fit_x.size() > 50) {
        auto ransac = ransac_affine_fit(fit_x, fit_y);
        inlier_ratios.reserve(ransac.num_inliers);
        inlier_pixel_idx.reserve(ransac.num_inliers);
        for (size_t j = 0; j < fit_idx.size(); ++j)
            if (ransac.inlier_mask[j]) {
                inlier_ratios.push_back(static_cast<double>(zo[fit_idx[j]]) / static_cast<double>(rd[fit_idx[j]]));
                inlier_pixel_idx.push_back(fit_idx[j]);
            }
    }

    // Compute s_global with MAD trim, track which pixels survived
    double s_global;
    std::vector<int> scale_pixel_idx;  // pixel indices that determined s_global

    if (inlier_ratios.size() >= 50) {
        double med = compute_median(inlier_ratios);
        double mad = compute_mad(inlier_ratios);
        if (mad > 0) {
            double trim = 3.5 * mad;
            for (size_t i = 0; i < inlier_ratios.size(); ++i) {
                if (std::abs(inlier_ratios[i] - med) <= trim)
                    scale_pixel_idx.push_back(inlier_pixel_idx[i]);
            }
            // Recompute median from trimmed set
            std::vector<double> trimmed;
            trimmed.reserve(scale_pixel_idx.size());
            for (size_t i = 0; i < inlier_ratios.size(); ++i) {
                if (std::abs(inlier_ratios[i] - med) <= trim)
                    trimmed.push_back(inlier_ratios[i]);
            }
            s_global = trimmed.empty() ? med : compute_median(trimmed);
        } else {
            s_global = med;
            scale_pixel_idx = inlier_pixel_idx;
        }
    } else {
        s_global = robust_median(ratios, config.global_trim_k);
        // No tracked pixel indices for fallback path
    }
    s_global = std::max(0.1, std::min(10.0, s_global));

    // Global variance: median V of the pixels that determined s_global
    float v_global = 1e-2f;
    if (has_vpx) {
        std::vector<float> v_vals;
        if (!scale_pixel_idx.empty()) {
            v_vals.reserve(scale_pixel_idx.size());
            for (int idx : scale_pixel_idx)
                if (std::isfinite(vp[idx])) v_vals.push_back(vp[idx]);
        }
        // Fallback: all valid pixels if not enough
        if (v_vals.size() < 50) {
            v_vals.clear();
            for (int i = 0; i < total; ++i)
                if (std::isfinite(vp[i])) v_vals.push_back(vp[i]);
        }
        if (!v_vals.empty()) {
            size_t mid = v_vals.size() / 2;
            std::nth_element(v_vals.begin(), v_vals.begin() + mid, v_vals.end());
            v_global = v_vals[mid];
        }
    }

    // === Per-label or global application ===
    float* z_out = result.z_out.ptr<float>(0);
    float* v_out = result.V_out.ptr<float>(0);

    if (!label_index.empty()) {
        #pragma omp parallel for schedule(dynamic)
        for (int L = 0; L < label_index.num_labels; ++L) {
            int n_px = label_index.count(L);
            if (n_px == 0) continue;
            const int* begin = label_index.begin(L);
            const int* end = label_index.end(L);

            // Collect ratios and pixel indices for this label
            std::vector<double> lr;
            std::vector<int> lr_idx;  // pixel indices corresponding to lr
            lr.reserve(n_px);
            lr_idx.reserve(n_px);
            for (const int* p = begin; p != end; ++p) {
                int idx = *p;
                if (std::isfinite(rd[idx]) && std::isfinite(zo[idx]) && rd[idx] > 0 && zo[idx] > 0 &&
                    (!has_mask || mk[idx] != 0)) {
                    lr.push_back(static_cast<double>(zo[idx]) / static_cast<double>(rd[idx]));
                    lr_idx.push_back(idx);
                }
            }

            // Determine scale
            float s_label = static_cast<float>(s_global);
            float v_label = v_global;
            float ratio_valid = (n_px > 0) ? static_cast<float>(lr.size()) / n_px : 0.0f;

            if (ratio_valid >= config.min_pts_ratio && !lr.empty()) {
                // MAD trim + track surviving pixel indices
                double med_L = compute_median(lr);
                double mad_L = compute_mad(lr);
                std::vector<double> trimmed_lr;
                std::vector<int> trimmed_idx;
                if (mad_L > 0) {
                    double trim = 3.5 * mad_L;
                    for (size_t i = 0; i < lr.size(); ++i) {
                        if (std::abs(lr[i] - med_L) <= trim) {
                            trimmed_lr.push_back(lr[i]);
                            trimmed_idx.push_back(lr_idx[i]);
                        }
                    }
                } else {
                    trimmed_lr = lr;
                    trimmed_idx = lr_idx;
                }
                s_label = static_cast<float>(trimmed_lr.empty() ? med_L : compute_median(trimmed_lr));

                // Variance from the pixels that determined s_label
                if (has_vpx && !trimmed_idx.empty()) {
                    std::vector<float> var_vals;
                    var_vals.reserve(trimmed_idx.size());
                    for (int idx : trimmed_idx)
                        if (std::isfinite(vp[idx])) var_vals.push_back(vp[idx]);
                    if (!var_vals.empty()) {
                        size_t vm = var_vals.size() / 2;
                        std::nth_element(var_vals.begin(), var_vals.begin() + vm, var_vals.end());
                        v_label = var_vals[vm];
                    }
                }
            }

            // Apply
            for (const int* p = begin; p != end; ++p) {
                z_out[*p] = s_label * rd[*p];
                v_out[*p] = v_label;
            }
        }
    } else {
        // No labels: global scale — vectorized Mat ops
        const float sg = static_cast<float>(s_global);

        // valid_rd: finite AND > 0 (NaN > 0 is false, handles both)
        cv::Mat valid_rd = (d_rel > 0);
        if (has_mask) valid_rd &= mask;

        // z_out = sg * d_rel where valid
        cv::Mat scaled;
        cv::multiply(d_rel, cv::Scalar(sg), scaled);
        scaled.copyTo(result.z_out, valid_rd);
        result.V_out.setTo(v_global, valid_rd);
    }


    // Fill NaN only where mask is valid (sky/invalid pixels stay NaN)
    // Vectorized: build valid output mask, find median, then fill
    cv::Mat z_valid_mask;
    {
        cv::Mat finite_mask;
        cv::compare(result.z_out, result.z_out, finite_mask, cv::CMP_EQ);  // NaN != NaN
        z_valid_mask = finite_mask & (result.z_out > 0);
    }

    float z_med = 1.0f;
    {
        std::vector<double> vz;
        vz.reserve(total / 4);
        for (int i = 0; i < total; ++i)
            if (std::isfinite(z_out[i]) && z_out[i] > 0) vz.push_back(z_out[i]);
        if (!vz.empty()) z_med = static_cast<float>(compute_median(vz));
    }

    // Fill invalid z_out pixels where mask allows
    cv::Mat fill_mask = ~z_valid_mask;
    if (has_mask) fill_mask &= mask;

    // Where z_obs is valid, use it; otherwise use z_med
    cv::Mat zo_valid = (z_obs > 0);
    cv::Mat zo_finite;
    cv::compare(z_obs, z_obs, zo_finite, cv::CMP_EQ);
    cv::Mat use_zo = fill_mask & zo_valid & zo_finite;
    cv::Mat use_med = fill_mask & ~(zo_valid & zo_finite);

    z_obs.copyTo(result.z_out, use_zo);
    result.z_out.setTo(z_med, use_med);


    return result;
}

// ============================================================================
// Public API
// ============================================================================

MetricScaleResult solve_metric_from_rel(
    const cv::Mat& d_rel, const cv::Mat& z_obs,
    const cv::Mat& mask, const LabelIndex& label_index,
    const cv::Mat& v_px, const MetricScaleConfig& config
) {
    if (d_rel.empty() || z_obs.empty()) return MetricScaleResult{};
    return solve_metric_core(d_rel, z_obs, mask, label_index, v_px, config);
}

}  // namespace ptc_depth
