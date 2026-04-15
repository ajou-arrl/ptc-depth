// Bayesian scale fusion
#pragma once

#include <opencv2/core.hpp>
#include "types.hpp"
#include "config.hpp"
#include "scale_estimation.hpp"

namespace ptc_depth {

class BayesianFusion {
public:
    struct Result {
        cv::Mat z_refined;      // Fused metric depth
        cv::Mat V_post;         // Posterior variance
        bool frame_rejected = false;
        int n_valid = 0;        // Pixels that entered Kalman update
        int n_bad = 0;          // Pixels exceeding chi2 hard gate
    };

    BayesianFusion(const CameraIntrinsics& cam, float max_depth);

    Result fuse(
        const cv::Mat& Z_prior,
        cv::Mat V_prior,
        const cv::Mat& Z_tri,
        const cv::Mat& rho,
        const cv::Mat& d_rel,
        float baseline,
        float b_ref,
        const FusionConfig& config
    );

    Result first_frame(
        const cv::Mat& Z_tri,
        const cv::Mat& rho,
        float baseline,
        float b_ref,
        const FusionConfig& config,
        int H, int W
    );

    void reset();

private:
    CameraIntrinsics cam_;
    float max_depth_;
    float f_eff_;   // sqrt(fx * fy), cached for inflate_V_prior

    // Temporal state
    float sigma_e_state_ = 0.1f;

    // Adaptive variance inflation using median Sampson residual
    void inflate_V_prior(cv::Mat& V_prior, const cv::Mat& rho,
                         float tau0_deg, float baseline, float b_ref);

    // Estimate frame-level consistency tolerance σ_e
    float update_sigma_e(const cv::Mat& Z_prior, const cv::Mat& Z_tri,
                         const cv::Mat& d_rel, const FusionConfig& config);

    // Per-pixel Kalman update in S-space
    Result kalman_update(
        const cv::Mat& Z_prior, const cv::Mat& V_prior,
        const cv::Mat& Z_tri, const cv::Mat& V_obs,
        const cv::Mat& d_rel, float sigma_e,
        const FusionConfig& config
    );
};
}  // namespace ptc_depth
