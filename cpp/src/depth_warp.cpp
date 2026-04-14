/**
 * Depth warp utility functions: rotation helpers and dense depth warping
 */
#include "ptc_depth/depth_warp.hpp"
#include "ptc_depth/types.hpp"
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace ptc_depth {

// Rodrigues: R → axis-angle ω (= axis * θ)
Eigen::Vector3d rotation_matrix_to_omega(const Eigen::Matrix3d& R) {
    double trace = R.trace();
    double cos_angle = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
    double angle = std::acos(cos_angle);

    if (angle < kEpsAngle) {
        // Small angle: ω ≈ vee(R - R^T) / 2
        return Eigen::Vector3d(
            (R(2,1) - R(1,2)) / 2.0,
            (R(0,2) - R(2,0)) / 2.0,
            (R(1,0) - R(0,1)) / 2.0
        );
    }

    if (angle > M_PI - kEpsAngle) {
        // Near π: axis from largest column of (R + I)
        Eigen::Matrix3d B = R + Eigen::Matrix3d::Identity();
        int max_col = 0;
        double max_norm = B.col(0).norm();
        for (int i = 1; i < 3; ++i) {
            double norm = B.col(i).norm();
            if (norm > max_norm) {
                max_norm = norm;
                max_col = i;
            }
        }
        Eigen::Vector3d axis = B.col(max_col).normalized();
        return axis * angle;
    }

    // General: axis = vee(R - R^T) / (2 sin θ)
    double sin_angle = std::sin(angle);
    Eigen::Vector3d axis(
        (R(2,1) - R(1,2)) / (2.0 * sin_angle),
        (R(0,2) - R(2,0)) / (2.0 * sin_angle),
        (R(1,0) - R(0,1)) / (2.0 * sin_angle)
    );

    return axis * angle;
}

// Rodrigues: axis-angle ω → R
Eigen::Matrix3d omega_to_R_cam(const Eigen::Vector3d& omega) {
    double angle = omega.norm();
    if (angle < kEpsAngle) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = omega / angle;
    Eigen::Matrix3d K;
    K <<        0, -axis(2),  axis(1),
         axis(2),        0, -axis(0),
        -axis(1),  axis(0),        0;
    return Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * K * K;
}

// Flow-based dense warp: depth + variance, Z-buffer keeps closest
void warp_dense_combined(
    const cv::Mat& prev_depth, const cv::Mat& prev_V,
    const cv::Mat& flow, const CameraIntrinsics& cam,
    float lambda_forget, float min_var,
    cv::Mat& z_warp_out, cv::Mat& V_warp_out
) {
    const int H = cam.H, W = cam.W;
    z_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    V_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (prev_depth.empty() || prev_V.empty() || flow.empty()) {
        return;
    }

    cv::Mat min_depth(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));

    for (int v0 = 0; v0 < H; ++v0) {
        const float* flow_row = flow.ptr<float>(v0);
        const float* depth_row = prev_depth.ptr<float>(v0);
        const float* V_row = prev_V.ptr<float>(v0);

        for (int u0 = 0; u0 < W; ++u0) {
            float z0 = depth_row[u0];
            float var = V_row[u0];

            float fx = flow_row[u0 * 2 + 0];
            float fy = flow_row[u0 * 2 + 1];

            int u1 = static_cast<int>(std::round(u0 + fx));
            int v1_coord = static_cast<int>(std::round(v0 + fy));
            if (u1 < 0 || u1 >= W || v1_coord < 0 || v1_coord >= H) continue;

            float* z_warp_target = z_warp_out.ptr<float>(v1_coord);
            float* V_warp_target = V_warp_out.ptr<float>(v1_coord);
            float* min_depth_target = min_depth.ptr<float>(v1_coord);

            if (std::isfinite(z0) && z0 > 1e-8f && z0 < min_depth_target[u1]) {
                min_depth_target[u1] = z0;
                z_warp_target[u1] = z0;
                if (std::isfinite(var)) {
                    V_warp_target[u1] = var * (1.0f + lambda_forget) + min_var;
                }
            }
        }
    }
}

void warp_prior_3d(
    const cv::Mat& prev_depth, const cv::Mat& prev_V,
    const Eigen::Matrix3d& R_tri, const Eigen::Vector3d& t_tri,
    const CameraIntrinsics& cam,
    float lambda_forget, float min_var,
    cv::Mat& z_warp_out, cv::Mat& V_warp_out
) {
    const int H = cam.H, W = cam.W;
    const float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;
    z_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    V_warp_out = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (prev_depth.empty()) {
        return;
    }

    Eigen::Matrix3d R_motion = R_tri.transpose();
    Eigen::Vector3d t_motion = -R_motion * t_tri;

    const double r00 = R_motion(0,0), r01 = R_motion(0,1), r02 = R_motion(0,2);
    const double r10 = R_motion(1,0), r11 = R_motion(1,1), r12 = R_motion(1,2);
    const double r20 = R_motion(2,0), r21 = R_motion(2,1), r22 = R_motion(2,2);
    const double tx = t_motion(0), ty = t_motion(1), tz = t_motion(2);

    cv::Mat min_depth(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
    bool has_V = !prev_V.empty();

    for (int v0 = 0; v0 < H; ++v0) {
        const float* depth_row = prev_depth.ptr<float>(v0);
        const float* V_row = has_V ? prev_V.ptr<float>(v0) : nullptr;

        for (int u0 = 0; u0 < W; ++u0) {
            float z0 = depth_row[u0];
            if (!std::isfinite(z0) || z0 <= 1e-8f) continue;

            double nx = (u0 - cx) / fx;
            double ny = (v0 - cy) / fy;
            double X0 = nx * z0;
            double Y0 = ny * z0;
            double Z0 = z0;

            double X1 = r00 * X0 + r01 * Y0 + r02 * Z0 + tx;
            double Y1 = r10 * X0 + r11 * Y0 + r12 * Z0 + ty;
            double Z1 = r20 * X0 + r21 * Y0 + r22 * Z0 + tz;

            if (Z1 <= 1e-8) continue;

            double u1d = fx * X1 / Z1 + cx;
            double v1d = fy * Y1 / Z1 + cy;

            int u1 = static_cast<int>(std::round(u1d));
            int v1 = static_cast<int>(std::round(v1d));

            if (u1 < 0 || u1 >= W || v1 < 0 || v1 >= H) continue;

            float z1_f = static_cast<float>(Z1);
            float* min_ptr = min_depth.ptr<float>(v1);
            if (z1_f < min_ptr[u1]) {
                min_ptr[u1] = z1_f;
                z_warp_out.ptr<float>(v1)[u1] = z1_f;

                if (has_V) {
                    float var = V_row[u0];
                    if (std::isfinite(var)) {
                        V_warp_out.ptr<float>(v1)[u1] = var * (1.0f + lambda_forget) + min_var;
                    }
                }
            }
        }
    }
}

}  // namespace ptc_depth
