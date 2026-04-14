#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include "ptc_depth/types.hpp"
#include "ptc_depth/config.hpp"

namespace ptc_depth {

// Contiguous per-label pixel index (offsets into flat_pixels)
struct LabelIndex {
    int num_labels = 0;
    std::vector<int> offsets;      // size: num_labels+1
    std::vector<int> flat_pixels;  // size: total_pixels (H*W for dense)

    // Get pixel indices for label L
    inline const int* begin(int L) const { return flat_pixels.data() + offsets[L]; }
    inline const int* end(int L) const { return flat_pixels.data() + offsets[L + 1]; }
    inline int count(int L) const { return offsets[L + 1] - offsets[L]; }
    inline bool empty() const { return num_labels == 0; }
};

LabelIndex build_label_index(const cv::Mat& labels, int num_labels = 0);

// Convert Sampson residual ρ(x) to observation variance 
cv::Mat rho_to_variance(
    const cv::Mat& rho,
    const CameraIntrinsics& cam,
    float baseline,
    float b_ref_auto,
    const FusionConfig& config
);

struct MetricScaleResult {
    cv::Mat z_out;      // Metric depth (HxW)
    cv::Mat V_out;      // Variance map (HxW)
};

// Solve metric scale from relative depth
MetricScaleResult solve_metric_from_rel(
    const cv::Mat& d_rel,
    const cv::Mat& z_obs,
    const cv::Mat& mask,
    const LabelIndex& label_index,
    const cv::Mat& v_px = cv::Mat(),
    const MetricScaleConfig& config = MetricScaleConfig()
);

}  // namespace ptc_depth
