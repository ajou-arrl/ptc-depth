#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "ptc_depth/types.hpp"

namespace ptc_depth {

struct TriangulationConfig {
    float fill_value = NAN;
};

struct TriangulationResult {
    cv::Mat z1_tri;       // Triangulated depth map (H, W) float32
    cv::Mat rho;      // Sampson error map (H, W) float32
    int num_valid = 0;
};

class Triangulator {
public:
    explicit Triangulator(const CameraIntrinsics& cam,
                          const TriangulationConfig& config = TriangulationConfig());

    TriangulationResult triangulate(
        const Correspondences& matches,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t
    );

private:
    CameraIntrinsics cam_;
    TriangulationConfig config_;
    Eigen::Matrix3d K_inv_;
};
}  // namespace ptc_depth
