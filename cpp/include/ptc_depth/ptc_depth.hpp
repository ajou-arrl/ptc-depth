#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "config.hpp"
#include "scale_estimation.hpp"
#include "bayesian_fusion.hpp"
#include "motion_field.hpp"
#include "triangulation.hpp"

namespace ptc_depth {


// Primary output from the pipeline.
struct Result {
    cv::Mat Z_post;          // Final metric depth (H, W) float32
    cv::Mat variance;        // Variance map (H, W) float32
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();  // Relative pose as 4×4 SE(3)
};

struct ScaleFusionResult {
    cv::Mat z_refined;     // Final metric depth
    cv::Mat variance;      // Final variance

    // Verbose fields (config.verbose == true)
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    cv::Mat z_obs;         // Triangulation result
    cv::Mat z_fused;       // Fusion result (before metric scale)
};

// Flow → Motion → Triangulation → Fusion → Metric Scale
class PTCDepth {
public:
    const PTCDepthConfig& config() const { return config_; }

    explicit PTCDepth(const PTCDepthConfig& config);

    struct TriangulationPose {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();      // Forward: p_curr = R·p_prev + t
        Eigen::Matrix4d pose_tri = Eigen::Matrix4d::Identity();  // For triangulation (= pose.inverse())
        Correspondences matches;
        int num_matches = 0;
        bool is_backward = false;
    };

    void reset();

    /** Process one frame (public API). */
    Result process(
        const cv::Mat& image,
        const cv::Mat& d_rel,
        float baseline,
        const cv::Mat& flow = cv::Mat(),
        const std::optional<Eigen::Matrix3d>& external_rotation = std::nullopt
    );

    // Overloads exposing internal ScaleFusionResult (verbose output)
    ScaleFusionResult refine(
        const cv::Mat& img,
        const cv::Mat& d_rel,
        float baseline,
        const cv::Mat& seg_labels = cv::Mat(),
        const cv::Mat& flow = cv::Mat()
    );

    ScaleFusionResult refine(
        const cv::Mat& img,
        const cv::Mat& d_rel,
        float baseline,
        const std::optional<Eigen::Matrix3d>& external_R,
        const std::optional<Eigen::Vector3d>& external_t = std::nullopt,
        const cv::Mat& seg_labels = cv::Mat(),
        const cv::Mat& flow = cv::Mat()
    );


private:
    PTCDepthConfig config_;

    // Previous frame state
    cv::Mat prev_img_;
    cv::Mat prev_d_rel_;
    cv::Mat prev_depth_;
    cv::Mat prev_V_;

    // Camera intrinsics (shared across modules)
    CameraIntrinsics cam_;

    // Bayesian fusion module — owns sigma_e EMA state
    BayesianFusion fusion_;

    // Baseline tracking
    BaselineAutoState baseline_state_;

    // Sub-modules (persistent, constructed once)
    MotionFieldEstimator motion_estimator_;
    Triangulator triangulator_;
    cv::Ptr<cv::DISOpticalFlow> flow_estimator_;

    // Helper methods
    cv::Mat compute_flow(const cv::Mat& img_prev, const cv::Mat& img_curr);

    // Baseline guard: skip triangulation when baseline is too small/large
    bool handle_baseline_guard(const cv::Mat& img, const cv::Mat& d_rel_f,
                                float baseline, int H, int W, ScaleFusionResult& result,
                                const cv::Mat& flow = cv::Mat());

    // RANSAC affine outlier filter on z_obs
    void filter_outliers(cv::Mat& z_obs, const cv::Mat& d_rel, int H, int W);

    // Apply metric scale estimation
    void apply_metric_scale(cv::Mat& z_refined, cv::Mat& V_post,
                            const cv::Mat& d_rel, const cv::Mat& seg_labels, int H, int W);

    void apply_metric_scale(cv::Mat& z_refined, cv::Mat& V_post,
                            const cv::Mat& rel_depth, const cv::Mat& sky_mask,
                            const LabelIndex& label_index, int H, int W);


    // Estimate motion from flow and depth.
    struct MotionResult {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();  // Forward: p_curr = R·p_prev + t
        Correspondences matches;
        int num_inliers = 0;
        bool success = false;
    };
    MotionResult estimate_motion(
        const cv::Mat& flow,
        const cv::Mat& d_rel,
        const cv::Mat& mask,
        const std::optional<Eigen::Vector3d>& known_omega = std::nullopt
    );

    struct TriResult {
        cv::Mat z_obs;
        cv::Mat rho;
        int num_valid = 0;
    };

    void update_state(
        const cv::Mat& img,
        const cv::Mat& d_rel,
        const cv::Mat& depth,
        const cv::Mat& V
    );

    // Helper structs and functions
    struct FlowSegResult {
        cv::Mat seg_labels;        // H x W, CV_32S (external or empty)
        int num_segments;
        cv::Mat flow_fwd;          // H x W x 2, float32
        cv::Mat flow_bwd;          // H x W x 2, float32 (or empty if iterative refinement disabled)
    };

    FlowSegResult prepare_flow_and_seg(
        const cv::Mat& img_cur,
        const cv::Mat& seg_labels_in,
        const cv::Mat& flow = cv::Mat()
    );


    struct TriWarpResult {
        TriResult tri;             // triangulation result
        cv::Mat z_prior_pose;      // 3D warp result
        cv::Mat V_warp_pose;       // warp variance
    };

    TriWarpResult compute_tri_and_warp(
        const Eigen::Matrix4d& pose_tri,
        const Correspondences& matches,
        bool has_prev_depth,
        const cv::Mat& prev_depth,
        const cv::Mat& prev_V,
        float lambda_forget,
        float min_var
    );

    // Validate external rotation → omega for motion estimation (std::nullopt if invalid/disabled)
    std::optional<Eigen::Vector3d> validate_external_rotation(
        const std::optional<Eigen::Matrix3d>& external_R) const;

    // Run motion estimation and convert to triangulation pose.
    // Returns std::nullopt if motion estimation fails.
    std::optional<TriangulationPose> setup_pose(
        bool use_backward,
        const cv::Mat& flow, const cv::Mat& d_rel, const cv::Mat& mask,
        float baseline,
        const std::optional<Eigen::Matrix3d>& external_R,
        const std::optional<Eigen::Vector3d>& external_t
    );

    // Warp prior selection: choose between pose warp and flow warp
    struct PriorSelection { cv::Mat z_prior; cv::Mat V_prior; };
    PriorSelection select_warp_prior(
        const cv::Mat& z_pose, const cv::Mat& V_pose,
        const cv::Mat& z_flow, const cv::Mat& V_flow, int H, int W) const;

    // Compute inverse depth from refined depth (for backward iteration)
    cv::Mat compute_inv_depth_from_refined(const cv::Mat& z_refined, int H, int W) const;

    // Normalize translation to baseline magnitude
    static Eigen::Vector3d normalize_t(const Eigen::Vector3d& t, float baseline);

};
}  // namespace ptc_depth
