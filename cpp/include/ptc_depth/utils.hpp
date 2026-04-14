// Shared statistical utilities (median, MAD, percentile, RANSAC affine fit)
#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <opencv2/core.hpp>
#include <cmath>
#include "ptc_depth/types.hpp"

namespace ptc_depth {

inline double compute_median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (n % 2 == 0) {
        double a = v[mid];
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        return (a + v[mid - 1]) / 2.0;
    }
    return v[mid];
}

inline double compute_mad(std::vector<double>& v) {
    double med = compute_median(v);
    std::vector<double> abs_dev(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        abs_dev[i] = std::abs(v[i] - med);
    return compute_median(abs_dev);
}

inline double compute_percentile(std::vector<double>& v, double pctl) {
    if (v.empty()) return 0.0;
    double idx = (pctl / 100.0) * (static_cast<double>(v.size()) - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo >= v.size()) lo = v.size() - 1;
    if (hi >= v.size()) hi = v.size() - 1;

    std::nth_element(v.begin(), v.begin() + lo, v.end());
    double val_lo = v[lo];

    if (lo == hi) return val_lo;

    std::nth_element(v.begin() + lo + 1, v.begin() + hi, v.end());
    double val_hi = v[hi];

    double frac = idx - static_cast<double>(lo);
    return val_lo * (1.0 - frac) + val_hi * frac;
}

inline double compute_mad_threshold(const std::vector<double>& values, double k = 3.5) {
    if (values.empty()) return 1e6;
    std::vector<double> v = values;
    double med = compute_median(v);
    std::vector<double> abs_dev(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        abs_dev[i] = std::abs(v[i] - med);
    double mad = compute_median(abs_dev);
    return med + k * mad + kEpsRange;
}

/**
 * Convert inverse depth to metric depth with validity mask.
 * inv_depth: (H,W) float32, values in [0,1] (0=far, 1=close)
 * depth_out: (H,W) float32, metric depth (1/inv_depth)
 * valid_out: (H,W) uint8 (optional), 1=valid, 0=invalid
 */
inline void inv_depth_to_depth(const cv::Mat& inv_depth,
                                cv::Mat& depth_out, cv::Mat& valid_out,
                                float invalid_fill = 100.0f) {
    const int H = inv_depth.rows, W = inv_depth.cols;
    depth_out.create(H, W, CV_32F);
    valid_out.create(H, W, CV_8U);
    for (int r = 0; r < H; ++r) {
        const float* inv_row = inv_depth.ptr<float>(r);
        float* d_row = depth_out.ptr<float>(r);
        uint8_t* v_row = valid_out.ptr<uint8_t>(r);
        for (int c = 0; c < W; ++c) {
            float val = inv_row[c];
            bool ok = std::isfinite(val) && val >= 1e-7f;
            v_row[c] = ok ? 1 : 0;
            d_row[c] = ok ? (1.0f / (val + kEpsDenomF)) : invalid_fill;
        }
    }
}

// ============================================================================
// RANSAC affine fit: y = a*x + b
// ============================================================================
struct RansacAffineResult {
    double a = 1.0, b = 0.0;
    int num_inliers = 0;
    std::vector<bool> inlier_mask;  // per-sample inlier flag
};

inline RansacAffineResult ransac_affine_fit(
    const std::vector<float>& x,
    const std::vector<float>& y,
    int max_iters = 100,
    double rel_thr = 0.2,
    uint32_t seed = 42
) {
    RansacAffineResult out;
    const int n_pts = static_cast<int>(x.size());
    if (n_pts < 2) return out;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n_pts - 1);

    const float thr_f = static_cast<float>(rel_thr);
    const int n_score = (n_pts + 1) / 2;  // points evaluated per iter (stride 2)
    const int early_stop = static_cast<int>(n_score * 0.9f);

    int best_inliers = 0;
    for (int iter = 0; iter < max_iters; ++iter) {
        int i1 = dist(rng), i2 = dist(rng);
        if (i1 == i2) continue;
        float dx = x[i2] - x[i1];
        if (std::abs(dx) < 1e-10f) continue;
        float a = (y[i2] - y[i1]) / dx;
        float b_val = y[i1] - a * x[i1];
        if (a <= 0.001f) continue;

        int inliers = 0;
        for (int j = 0; j < n_pts; j += 2) {
            float pred = a * x[j] + b_val;
            // |pred - y| < thr * y  (avoids division, valid since y > 0)
            if (pred > 1e-6f && std::abs(pred - y[j]) < thr_f * y[j])
                inliers++;
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            out.a = a;
            out.b = b_val;
            if (best_inliers >= early_stop) break;
        }
    }

    // Build full inlier mask
    out.inlier_mask.resize(n_pts, false);
    out.num_inliers = 0;
    const float a_f = static_cast<float>(out.a);
    const float b_f = static_cast<float>(out.b);
    for (int j = 0; j < n_pts; ++j) {
        float pred = a_f * x[j] + b_f;
        if (pred > 1e-6f && std::abs(pred - y[j]) < thr_f * y[j]) {
            out.inlier_mask[j] = true;
            out.num_inliers++;
        }
    }

    return out;
}
} // namespace ptc_depth
