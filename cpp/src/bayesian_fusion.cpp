/**
 * Bayesian scale fusion implementation (Paper §3.4, Eq. 7-11)
 */
#include "ptc_depth/bayesian_fusion.hpp"
#include "ptc_depth/config.hpp"
#include "ptc_depth/utils.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace ptc_depth {

BayesianFusion::BayesianFusion(const CameraIntrinsics& cam, float max_depth)
    : cam_(cam), max_depth_(max_depth),
      f_eff_(std::sqrt(std::max(cam.fx, kEpsDenomF) * std::max(cam.fy, kEpsDenomF))) {}

void BayesianFusion::reset() {
    sigma_e_state_ = 0.1f;
}


// First frame: no prior available, pass through observation
BayesianFusion::Result BayesianFusion::first_frame(
    const cv::Mat& Z_tri,
    const cv::Mat& rho,
    float baseline,
    float b_ref,
    const FusionConfig& config,
    int H, int W
) {
    Result out;
    out.frame_rejected = false;

    cv::Mat V_obs;
    if (rho.empty() || Z_tri.empty() || rho.rows != H || rho.cols != W) {
        V_obs = cv::Mat(H, W, CV_32F, cv::Scalar(100.0f));
    } else {
        V_obs = rho_to_variance(
            rho, cam_, baseline, b_ref, config
        );
    }

    out.z_refined = Z_tri.clone();
    out.V_post = V_obs;
    return out;
}

BayesianFusion::Result BayesianFusion::fuse(
    const cv::Mat& Z_prior,
    cv::Mat V_prior,
    const cv::Mat& Z_tri,
    const cv::Mat& rho,
    const cv::Mat& d_rel,
    float baseline,
    float b_ref,
    const FusionConfig& config
) {
    const int H = Z_prior.rows;
    const int W = Z_prior.cols;

    // --- Compute V_obs from Sampson residual ---
    cv::Mat V_obs = rho_to_variance(
        rho, cam_, baseline, b_ref, config
    );

    // --- Adaptive variance inflation ---
    inflate_V_prior(V_prior, rho, config.tau0_deg, baseline, b_ref);

    // --- Estimate σ_e ---
    float sigma_e = update_sigma_e(Z_prior, Z_tri, d_rel, config);


    // --- Per-pixel Kalman update ---
    Result out = kalman_update(Z_prior, V_prior, Z_tri, V_obs, d_rel, sigma_e, config);

    // --- Frame rejection decision ---
    out.frame_rejected = false;
    if (config.frame_reject_enable && out.n_valid >= config.frame_reject_min_valid) {
        float bad_frac = static_cast<float>(out.n_bad) / out.n_valid;
        if (bad_frac >= config.frame_reject_bad_frac) {
            out.frame_rejected = true;
        }
    }

    if (out.frame_rejected) {
        Z_prior.copyTo(out.z_refined);
        V_prior.copyTo(out.V_post);
        out.V_post *= 1.5f;
    }

    return out;
}

void BayesianFusion::inflate_V_prior(
    cv::Mat& V_prior, const cv::Mat& rho,
    float tau0_deg, float baseline, float b_ref
) {
    if (rho.empty()) return;

    const int H = rho.rows;
    const int W = rho.cols;


    std::vector<float> rho_vals;
    rho_vals.reserve(H * W / 16);
    for (int r = 0; r < H; r += 4) {
        const float* rho_row = rho.ptr<float>(r);
        for (int c = 0; c < W; c += 4) {
            float val = rho_row[c];
            if (std::isfinite(val) && val > 0) {
                rho_vals.push_back(val);
            }
        }
    }

    if (rho_vals.size() <= 10) return;

    size_t mid = rho_vals.size() / 2;
    std::nth_element(rho_vals.begin(), rho_vals.begin() + mid, rho_vals.end());
    float rho_bar = rho_vals[mid];

    float r_ang = std::sqrt(rho_bar) / f_eff_;

    float tau0 = (tau0_deg > 0) ? tau0_deg * static_cast<float>(M_PI) / 180.0f : 1.75e-3f;
    // Scale tau0 by baseline ratio (same as rho_to_variance)
    float B = (std::isfinite(baseline) && baseline > 0) ? baseline : b_ref;
    float bref = (std::isfinite(b_ref) && b_ref > 0) ? b_ref : 1.0f;
    tau0 *= (B / bref);
    float ratio = r_ang / tau0;
    float inflation = 1.0f + ratio * ratio;

    if (inflation > 1.001f) {
        V_prior = V_prior.clone();
        V_prior *= inflation;
    }
}

float BayesianFusion::update_sigma_e(
    const cv::Mat& Z_prior, const cv::Mat& Z_tri,
    const cv::Mat& d_rel, const FusionConfig& config
) {
    const int H = Z_prior.rows;
    const int W = Z_prior.cols;
    const int step = 4;

    std::vector<float> delta_vals;
    delta_vals.reserve(H * W / (step * step));

    for (int r = 0; r < H; r += step) {
        const float* zw_row = Z_prior.ptr<float>(r);
        const float* zt_row = Z_tri.ptr<float>(r);
        const float* lam_row = d_rel.ptr<float>(r);
        for (int c = 0; c < W; c += step) {
            float lam = lam_row[c];
            if (std::isfinite(lam) && lam > 1e-8f) {
                float zw = zw_row[c];
                float zt = zt_row[c];
                if (std::isfinite(zw) && zw > 1e-8f && std::isfinite(zt) && zt > 1e-8f) {
                    float S_prior = zw * lam;
                    float S_obs   = zt * lam;
                    float delta = std::abs(S_prior - S_obs) / std::max(S_obs, kEpsDenomF);
                    delta_vals.push_back(delta);
                }
            }
        }
    }

    float sigma_e_raw = sigma_e_state_;
    if (!delta_vals.empty()) {
        size_t n = delta_vals.size();
        size_t mid = n / 2;
        std::nth_element(delta_vals.begin(), delta_vals.begin() + mid, delta_vals.end());
        float median = delta_vals[mid];

        for (auto& v : delta_vals) {
            v = std::abs(v - median);
        }
        std::nth_element(delta_vals.begin(), delta_vals.begin() + mid, delta_vals.end());
        float mad = delta_vals[mid];

        sigma_e_raw = std::max(1.4826f * mad, config.sigma_e_min);
    }

    sigma_e_state_ = (1.0f - config.sigma_e_ema_beta) * sigma_e_state_ +
                     config.sigma_e_ema_beta * sigma_e_raw;

    return std::clamp(sigma_e_state_, config.sigma_e_min, config.sigma_e_max);
}

BayesianFusion::Result BayesianFusion::kalman_update(
    const cv::Mat& Z_prior, const cv::Mat& V_prior,
    const cv::Mat& Z_tri, const cv::Mat& V_obs,
    const cv::Mat& d_rel, float sigma_e,
    const FusionConfig& config
) {
    const int H = Z_prior.rows;
    const int W = Z_prior.cols;

    Result out;
    out.z_refined = cv::Mat(H, W, CV_32F);  // every pixel written in loop below
    out.V_post    = cv::Mat(H, W, CV_32F);  // every pixel written in loop below
    out.frame_rejected = false;
    out.n_valid = 0;
    out.n_bad = 0;

    int n_valid = 0, n_bad = 0;

    #pragma omp parallel for reduction(+:n_valid, n_bad) schedule(static)
    for (int r = 0; r < H; ++r) {
        const float* z_warp_row = Z_prior.ptr<float>(r);
        const float* z_obs_row  = Z_tri.ptr<float>(r);
        const float* vp_row     = V_prior.ptr<float>(r);
        const float* vo_row     = V_obs.ptr<float>(r);
        const float* id_row     = d_rel.ptr<float>(r);
        float* zr_row  = out.z_refined.ptr<float>(r);
        float* vp_out  = out.V_post.ptr<float>(r);

        for (int c = 0; c < W; ++c) {
            float inv_depth = id_row[c];
            float vp = vp_row[c];
            float vo = vo_row[c];

            // Compute S = z · d_rel (scale-space)
            float S_prior = -1.0f, S_obs = -1.0f;
            if (std::isfinite(inv_depth) && inv_depth > 1e-8f) {
                float zw = z_warp_row[c];
                float zt = z_obs_row[c];
                if (std::isfinite(zw) && zw > 1e-8f) S_prior = zw * inv_depth;
                if (std::isfinite(zt) && zt > 1e-8f) S_obs   = zt * inv_depth;
            }

            // Consistency score: c(x) ≈ exp(-x²/2) (Eq. 21)
            // Padé approx: 1/(1 + x²/2 + x⁴/8 + x⁶/48)
            float consistency = 0.0f;
            if (S_prior > 0 && S_obs > 1e-8f) {
                float delta = std::abs(S_prior - S_obs) / std::max(S_obs, kEpsDenomF);
                float x = delta / sigma_e;
                float x2 = x * x;
                if (x2 < 18.0f) {
                    float x4 = x2 * x2;
                    consistency = 1.0f / (1.0f + 0.5f * x2 + 0.125f * x4 + x4 * x2 / 48.0f);
                }
            }

            bool sp_valid = S_prior > 0;
            bool so_valid = S_obs > 0;
            bool vp_valid = std::isfinite(vp) && vp > 0;
            bool vo_valid = std::isfinite(vo) && vo > 0;
            bool id_valid = std::isfinite(inv_depth) && inv_depth > 1e-8f;

            if (!sp_valid && !so_valid) {
                // Case 1: no valid data
                zr_row[c] = std::numeric_limits<float>::quiet_NaN();
                vp_out[c] = 100.0f;
            } else if (!sp_valid) {
                // Case 2: observation only
                vp_out[c] = vo_valid ? vo : 100.0f;
                zr_row[c] = id_valid ? (S_obs / inv_depth) : std::numeric_limits<float>::quiet_NaN();
            } else if (!so_valid) {
                // Case 3: prior only
                float v_inc = (vp_valid ? vp : 10.0f) * (1.0f + config.lambda_forget);
                vp_out[c] = std::min(std::max(v_inc, config.min_var), config.max_var);
                zr_row[c] = id_valid ? (S_prior / inv_depth) : std::numeric_limits<float>::quiet_NaN();
            } else {
                // Case 4: both valid → Kalman update
                float vp_eff = std::max(vp_valid ? vp : 10.0f, config.min_var);
                float vo_eff = std::max(vo, config.min_var);

                float innov = S_obs - S_prior;
                float V_innov = vp_eff + vo_eff;
                float gamma = (innov * innov) / V_innov;  // Eq. 8

                float chi2_hard_eff = config.chi2_hard *
                    (1.0f + 0.7f * config.gate_loosen * (1.0f - consistency));

                if (config.frame_reject_enable) {
                    n_valid++;
                    if (gamma > chi2_hard_eff) n_bad++;
                }

                // κ = min(κ_raw, κ_min + (1-κ_min)·c)
                float kappa_cap = config.kappa_min + (1.0f - config.kappa_min) * consistency;
                float kappa_raw = vp_eff / V_innov;   // Eq. 9
                float kappa = std::min(kappa_raw, kappa_cap);

                float S_post_val, v_post_val;
                if (gamma > chi2_hard_eff) {
                    // Hard rejection: keep lower-variance estimate
                    if (vp_eff < vo_eff) {
                        S_post_val = S_prior; v_post_val = vp_eff;
                    } else {
                        S_post_val = S_obs; v_post_val = vo_eff;
                    }
                } else {
                    // S_post = S_prior + κ·(S_obs - S_prior)
                    S_post_val = S_prior + kappa * innov;
                    v_post_val = std::max(
                        (1.0f - kappa) * (1.0f - kappa) * vp_eff + kappa * kappa * vo_eff,
                        config.min_var);
                }

                vp_out[c] = std::min(v_post_val, config.max_var);
                if (id_valid) {
                    zr_row[c] = std::min(S_post_val / inv_depth, max_depth_);
                } else {
                    zr_row[c] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }

    out.n_valid = n_valid;
    out.n_bad = n_bad;
    return out;
}

}  // namespace ptc_depth
