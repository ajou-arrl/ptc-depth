#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include "types.hpp"

namespace ptc_depth {

Eigen::Vector3d rotation_matrix_to_omega(const Eigen::Matrix3d& R);
Eigen::Matrix3d omega_to_R_cam(const Eigen::Vector3d& omega);

void warp_dense_combined(
    const cv::Mat& prev_depth, const cv::Mat& prev_V,
    const cv::Mat& flow, const CameraIntrinsics& cam,
    float lambda_forget, float min_var,
    cv::Mat& z_warp_out, cv::Mat& V_warp_out
);

void warp_prior_3d(
    const cv::Mat& prev_depth, const cv::Mat& prev_V,
    const Eigen::Matrix3d& R_tri, const Eigen::Vector3d& t_tri,
    const CameraIntrinsics& cam,
    float lambda_forget, float min_var,
    cv::Mat& z_warp_out, cv::Mat& V_warp_out
);
}  // namespace ptc_depth
