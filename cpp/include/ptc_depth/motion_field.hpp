// Motion field estimation (Eq. 1-2): ṗ = B·Ω + (d_rel)·A·T
#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <optional>
#include "ptc_depth/types.hpp"
#include "ptc_depth/config.hpp"

namespace ptc_depth {

struct MotionFieldResult {
    Eigen::Vector3d omega;               // ω (axis-angle, camera frame)
    Eigen::Vector3d T_hat;               // T̂ — unit translation direction

    cv::Mat inlier_mask;                 // (H,W) inlier mask
    Correspondences matches;             // Inlier correspondences
    int num_inliers = 0;
};

/**
 * Compute B and A matrices for motion field equation
 *
 * ṗ = B·ω + d_rel·A·t
 *
 * Given normalized image coordinates (x, y):
 *   B = [[x*y,      -(1+x²), y ],    (angular velocity term)
 *        [(1+y²),   -x*y,    -x]]
 *
 *   A = [[-1, 0, x],                  (translational velocity term)
 *        [ 0,-1, y]]
 */
inline void compute_motion_matrices(double nx, double ny,
                                    Eigen::Matrix<double, 2, 3>& B,
                                    Eigen::Matrix<double, 2, 3>& A) {
    // B matrix (rotation term, Eq. 1)
    B(0, 0) = nx * ny;
    B(0, 1) = -(1.0 + nx * nx);
    B(0, 2) = ny;

    B(1, 0) = (1.0 + ny * ny);
    B(1, 1) = -(nx * ny);
    B(1, 2) = -nx;

    // A matrix (translation term, Eq. 1)
    A(0, 0) = -1.0;
    A(0, 1) =  0.0;
    A(0, 2) =  nx;

    A(1, 0) =  0.0;
    A(1, 1) = -1.0;
    A(1, 2) =  ny;
}

class MotionFieldEstimator {
public:
    MotionFieldEstimator(const MotionFieldConfig& cfg,
                         const CameraIntrinsics& intrinsics);

    MotionFieldResult estimate(const cv::Mat& flow,
                               const cv::Mat& d_rel,
                               const cv::Mat& mask = cv::Mat(),
                               const std::optional<Eigen::Vector3d>& known_omega = std::nullopt);

private:
    MotionFieldConfig cfg_;
    double fx_, fy_, inv_fx_, inv_fy_, cx_, cy_;

    // all_* = full valid points (for classify_inliers), non-prefixed = subsampled (for RANSAC)
    struct CollectedPoints {
        // All valid points before subsampling (used for post-RANSAC inlier classification)
        std::vector<Eigen::Vector2d> all_flow_normalized;    // optical flow in normalized image coords
        std::vector<Eigen::Vector2d> all_coords_normalized;  // pixel position in normalized image coords
        std::vector<double> all_d_rels;                      // relative inverse depth
        Correspondences all_matches;                         // pixel correspondences (u0,v0 → u1,v1)

        // Subsampled points for RANSAC input (stratified by depth bin × spatial cell)
        std::vector<Eigen::Vector2d> flow_normalized;        // optical flow in normalized image coords
        std::vector<Eigen::Vector2d> coords_normalized;      // pixel position in normalized image coords
        std::vector<double> d_rels;                          // relative inverse depth
        std::vector<int> cell_ids;                           // spatial grid cell index
        std::vector<double> flow_mags;                       // flow magnitude in pixels (for RANSAC scoring)

        int max_cells = 0;  // total number of spatial grid cells (grid_rows * grid_cols)
    };

    /** Collect and subsample valid points from flow/depth images */
    CollectedPoints collect_points(
        const cv::Mat& flow,
        const cv::Mat& d_rel,
        const cv::Mat& mask);

    /** Depth-stratified subsampling: replaces pts sampled vectors in-place */
    static void depth_subsample(CollectedPoints& pts, const MotionFieldConfig& cfg);

    /** Compute flow pixel magnitudes into pts.flow_mags */
    static void pixel_norms(CollectedPoints& pts, double fx, double fy, float min_flow_px);

    /** Compute residual, angle error, and observed magnitude for point i */
    void residual_at(int i, const CollectedPoints& pts,
                     const Eigen::Vector3d& omega, const Eigen::Vector3d& t_cam,
                     double& out_residual, double& out_angle, double& out_obs_mag) const;

    /** Classify all points as inliers/outliers using RANSAC result */
    MotionFieldResult classify_inliers(
        const Eigen::Vector3d& omega,
        const Eigen::Vector3d& t_cam,
        const CollectedPoints& points,
        int H, int W);

};
} // namespace ptc_depth
