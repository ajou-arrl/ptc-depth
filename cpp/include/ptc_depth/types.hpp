#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <deque>

namespace ptc_depth {

constexpr double kEpsDenom  = 1e-12;  // Prevent division by zero (double: norm, MAD, ratio, etc.)
constexpr double kEpsRange  = 1e-6;   // Range/relative-error denominator guard (double)
constexpr float  kEpsDenomF = 1e-6f;  // Prevent division by zero (float)
constexpr double kEpsAngle  = 1e-10;  // Axis-angle singularity detection threshold

// Pixel correspondences between two frames
struct Correspondences {
    std::vector<float> u0, v0;  // Source frame pixel coordinates
    std::vector<float> u1, v1;  // Target frame pixel coordinates

    int size() const { return static_cast<int>(u0.size()); }
    bool empty() const { return u0.empty(); }
};

// Camera intrinsics (resolution + pinhole model)
struct CameraIntrinsics {
    int H = 0, W = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;

    CameraIntrinsics() = default;
    CameraIntrinsics(int h, int w, float fx_, float fy_, float cx_, float cy_)
        : H(h), W(w), fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    // Build 3×3 intrinsic matrix K
    Eigen::Matrix3d K() const {
        Eigen::Matrix3d m;
        m << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        return m;
    }
    // Analytical inverse (pinhole model is trivially invertible)
    Eigen::Matrix3d K_inv() const {
        double ifx = 1.0 / fx, ify = 1.0 / fy;
        Eigen::Matrix3d m;
        m << ifx, 0, -cx * ifx,
             0, ify, -cy * ify,
             0,   0,         1;
        return m;
    }
};

class BaselineAutoState {
public:
    explicit BaselineAutoState(float ema_beta = 0.5f, int hist_len = 100);

    bool check(float baseline, float min_baseline);
    float b_ref() const;
    bool has_ema() const { return has_ema_; }

private:
    void update(float b);

    float b_ema_;              // EMA of baseline magnitude
    bool has_ema_;             // whether EMA has been initialized
    std::deque<float> hist_;   // sliding window history for percentile-based b_ref (deque for O(1) pop_front)
    float beta_;               // EMA decay factor (0=instant, 1=no update)
    int hist_len_;             // max history length
};
} // namespace ptc_depth
