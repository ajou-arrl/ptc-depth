/**
 * PTC-Depth pipeline implementation
 */
#include "ptc_depth/ptc_depth.hpp"
#include "ptc_depth/depth_warp.hpp"
#include "ptc_depth/scale_estimation.hpp"
#include "ptc_depth/utils.hpp"
#include <cmath>
#include <algorithm>

namespace ptc_depth {


static Eigen::Matrix4d make_pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;
}

// PTCDepth Implementation
PTCDepth::PTCDepth(const PTCDepthConfig& config_in)
    : config_(config_in),
      cam_(config_in.H, config_in.W, config_in.fx, config_in.fy, config_in.cx, config_in.cy),
      fusion_(cam_, config_in.max_depth),
      baseline_state_(config_in.baseline_ema_beta, config_in.baseline_hist_len),
      motion_estimator_(config_in.motion, cam_),
      triangulator_(cam_,
                    TriangulationConfig{config_in.fill_value}) {
    flow_estimator_ = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
    flow_estimator_->setFinestScale(0);
    flow_estimator_->setUseSpatialPropagation(true);
}

void PTCDepth::reset() {
    prev_img_.release();
    prev_d_rel_.release();
    prev_depth_.release();
    prev_V_.release();
    fusion_.reset();
    baseline_state_ = BaselineAutoState(config_.baseline_ema_beta, config_.baseline_hist_len);
}

cv::Mat PTCDepth::compute_flow(const cv::Mat& img_prev, const cv::Mat& img_curr) {
    cv::Mat gray_prev, gray_curr;
    if (img_prev.channels() == 3) cv::cvtColor(img_prev, gray_prev, cv::COLOR_BGR2GRAY);
    else gray_prev = img_prev;
    if (img_curr.channels() == 3) cv::cvtColor(img_curr, gray_curr, cv::COLOR_BGR2GRAY);
    else gray_curr = img_curr;
    cv::Mat flow;
    flow_estimator_->calc(gray_prev, gray_curr, flow);
    return flow;
}

PTCDepth::MotionResult PTCDepth::estimate_motion(
    const cv::Mat& flow,
    const cv::Mat& d_rel,
    const cv::Mat& mask,
    const std::optional<Eigen::Vector3d>& known_omega
) {
    MotionResult result;
    result.success = false;
    result.num_inliers = 0;

    try {
        MotionFieldResult mf_result = motion_estimator_.estimate(flow, d_rel, mask, known_omega);

        // Convert (omega, T_hat) → 4×4 pose (point-transform convention)
        Eigen::Matrix3d R_cam = omega_to_R_cam(mf_result.omega);
        Eigen::Matrix3d R = R_cam.transpose();
        Eigen::Vector3d t = -(R * mf_result.T_hat);  // unit direction, baseline applied later
        result.pose = make_pose(R, t);
        result.matches = std::move(mf_result.matches);
        result.num_inliers = mf_result.num_inliers;
        result.success = (mf_result.num_inliers >= 50);

    } catch (const std::exception& e) {
        result.success = false;
    }

    return result;
}


void PTCDepth::update_state(
    const cv::Mat& img,
    const cv::Mat& d_rel,
    const cv::Mat& depth,
    const cv::Mat& V
) {
    img.copyTo(prev_img_);
    d_rel.copyTo(prev_d_rel_);
    depth.copyTo(prev_depth_);
    V.copyTo(prev_V_);
}

// Helper: RANSAC affine outlier filter on z_obs
void PTCDepth::filter_outliers(cv::Mat& z_obs, const cv::Mat& d_rel, int H, int W) {
    // Collect samples: x = inv_depth, y = 1/z_obs
    std::vector<float> fit_x, fit_y;
    std::vector<int> fit_idx;
    fit_x.reserve(H * W / 4);
    fit_y.reserve(H * W / 4);
    fit_idx.reserve(H * W / 4);

    const float* zo_data = z_obs.ptr<float>(0);
    const float* lam_data = d_rel.ptr<float>(0);
    int total = H * W;

    for (int i = 0; i < total; i += 2) {
        float zo = zo_data[i];
        float lam = lam_data[i];
        if (std::isfinite(zo) && zo > 0.1f && zo < config_.max_depth &&
            std::isfinite(lam) && lam > 1e-6f) {
            fit_x.push_back(lam);
            fit_y.push_back(1.0f / zo);
            fit_idx.push_back(i);
        }
    }

    if (fit_x.size() <= 50) return;

    auto result = ransac_affine_fit(fit_x, fit_y);

    // NaN-out outliers
    int n_removed = 0;
    float* zo_mut = z_obs.ptr<float>(0);
    for (size_t j = 0; j < fit_idx.size(); ++j) {
        if (!result.inlier_mask[j]) {
            zo_mut[fit_idx[j]] = std::numeric_limits<float>::quiet_NaN();
            n_removed++;
        }
    }

}

// Helper: apply metric scale estimation
void PTCDepth::apply_metric_scale(cv::Mat& z_refined, cv::Mat& V_post,
                                      const cv::Mat& d_rel, const cv::Mat& seg_labels,
                                      int H, int W) {

    // Convert d_rel (inverse depth) to rel_depth (proportional to metric depth)
    cv::Mat rel_depth, valid_mask;
    inv_depth_to_depth(d_rel, rel_depth, valid_mask);

    LabelIndex label_index = build_label_index(seg_labels, 0);
    apply_metric_scale(z_refined, V_post, rel_depth, valid_mask, label_index, H, W);
}

void PTCDepth::apply_metric_scale(cv::Mat& z_refined, cv::Mat& V_post,
                                      const cv::Mat& rel_depth, const cv::Mat& sky_mask,
                                      const LabelIndex& label_index, int H, int W) {

    const cv::Mat& mask = sky_mask;

    const auto& metric_cfg = config_.metric;

    MetricScaleResult metric_result = solve_metric_from_rel(
        rel_depth, z_refined, mask, label_index, V_post, metric_cfg
    );

    z_refined = metric_result.z_out;
    V_post = metric_result.V_out;

}

// Helper: baseline guard
bool PTCDepth::handle_baseline_guard(const cv::Mat& img, const cv::Mat& d_rel_f,
                                         float baseline, int H, int W,
                                         ScaleFusionResult& result,
                                         const cv::Mat& flow) {
    bool triggered = baseline_state_.check(baseline, config_.min_baseline);

    if (!triggered)
        return false;  // No guard triggered

    // Guard triggered — fill result with warp-only depth
    result.z_obs = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    if (!prev_depth_.empty() && !prev_img_.empty()) {
        cv::Mat flow_fwd = flow.empty() ? compute_flow(prev_img_, img) : flow;
        cv::Mat z_prior_flow, V_warp_flow;
        warp_dense_combined(prev_depth_, prev_V_, flow_fwd, cam_,
                            0.0f, config_.fusion.min_var,  // no variance inflation during guard
                            z_prior_flow, V_warp_flow);

        z_prior_flow.copyTo(result.z_refined);

        apply_metric_scale(result.z_refined, V_warp_flow, d_rel_f, cv::Mat(), H, W);

        // Sky pixels → inf + max_var
        if (config_.outdoor) {
            cv::Mat dummy, sky_mask;
            inv_depth_to_depth(d_rel_f, dummy, sky_mask);
            cv::Mat sky_inv = (sky_mask == 0);
            result.z_refined.setTo(std::numeric_limits<float>::infinity(), sky_inv);
            V_warp_flow.setTo(config_.fusion.max_var, sky_inv);
        }

        result.variance = V_warp_flow.clone();
        update_state(img, d_rel_f, result.z_refined, V_warp_flow);
    } else {
        result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        result.variance = cv::Mat(H, W, CV_32F, cv::Scalar(config_.fusion.max_var));
        prev_img_ = img.clone();
        prev_d_rel_ = d_rel_f.clone();
    }

    return true;  // Guard triggered, caller should return
}

PTCDepth::FlowSegResult PTCDepth::prepare_flow_and_seg(
    const cv::Mat& img_cur,
    const cv::Mat& seg_labels_in,
    const cv::Mat& flow
) {
    FlowSegResult out;
    out.num_segments = 0;

    if (!seg_labels_in.empty()) {
        out.seg_labels = seg_labels_in;
        double minVal, maxVal;
        cv::minMaxLoc(seg_labels_in, &minVal, &maxVal);
        out.num_segments = static_cast<int>(maxVal) + 1;
    }

    out.flow_fwd = flow.empty() ? compute_flow(prev_img_, img_cur) : flow;
    // flow_bwd computed lazily in refine() only when iter > 0 actually runs

    return out;
}


PTCDepth::TriWarpResult PTCDepth::compute_tri_and_warp(
    const Eigen::Matrix4d& pose_tri,
    const Correspondences& matches,
    bool has_prev_depth,
    const cv::Mat& prev_depth,
    const cv::Mat& prev_V,
    float lambda_forget,
    float min_var
) {
    TriWarpResult result;
    Eigen::Matrix3d R = pose_tri.block<3,3>(0,0);
    Eigen::Vector3d t = pose_tri.block<3,1>(0,3);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (!matches.empty()) {
                auto tri = triangulator_.triangulate(matches, R, t);
                result.tri.z_obs = tri.z1_tri;
                result.tri.rho = tri.rho;
                result.tri.num_valid = tri.num_valid;
            } else {
                result.tri.z_obs = cv::Mat(config_.H, config_.W, CV_32F, cv::Scalar(config_.fill_value));
                result.tri.rho = cv::Mat(config_.H, config_.W, CV_32F, cv::Scalar(config_.fill_value));
            }
        }

        #pragma omp section
        {
            if (has_prev_depth) {
                warp_prior_3d(prev_depth, prev_V, R, t, cam_,
                              lambda_forget, min_var,
                              result.z_prior_pose, result.V_warp_pose);
            }
        }
    }

    return result;
}

// 4-param version: calls 6-param version with no GT pose
// ============================================================================
// process(): Clean public API — returns Result (Z_post, confidence, pose)
// ============================================================================
Result PTCDepth::process(
    const cv::Mat& image,
    const cv::Mat& d_rel,
    float baseline,
    [[maybe_unused]] const cv::Mat& flow,
    const std::optional<Eigen::Matrix3d>& external_rotation
) {
    auto internal = refine(image, d_rel, baseline, external_rotation);

    // Convert to clean Result
    Result out;
    out.Z_post = std::move(internal.z_refined);
    out.variance = std::move(internal.variance);
    out.pose = internal.pose;

    return out;
}

// ============================================================================
// refine(): Internal API (backward-compatible, used by bindings)
// ============================================================================
ScaleFusionResult PTCDepth::refine(
    const cv::Mat& img,
    const cv::Mat& d_rel,
    float baseline,
    const cv::Mat& seg_labels,
    const cv::Mat& flow
) {
    return refine(img, d_rel, baseline, std::nullopt, std::nullopt, seg_labels, flow);
}

ScaleFusionResult PTCDepth::refine(
    const cv::Mat& img,
    const cv::Mat& d_rel,
    float baseline,
    const std::optional<Eigen::Matrix3d>& external_R,
    const std::optional<Eigen::Vector3d>& external_t,
    const cv::Mat& seg_labels,
    const cv::Mat& flow
) {
    ScaleFusionResult result;

    const int H = config_.H;
    const int W = config_.W;

    if (img.rows != H || img.cols != W)
        throw std::runtime_error("Image size mismatch: expected " +
            std::to_string(H) + "x" + std::to_string(W));
    if (d_rel.rows != H || d_rel.cols != W)
        throw std::runtime_error("d_rel size mismatch: expected " +
            std::to_string(H) + "x" + std::to_string(W));

    const cv::Mat& img_resized = img;
    cv::Mat d_rel_f;
    if (d_rel.type() == CV_32F) d_rel_f = d_rel;
    else d_rel.convertTo(d_rel_f, CV_32F);

    cv::Mat rel_depth, sky_mask;
    inv_depth_to_depth(d_rel_f, rel_depth, sky_mask);
    // sky_mask: 1 = valid depth, 0 = sky/invalid (d_rel ≈ 0)

    // First frame: store state and return NaN (no previous frame for optical flow)
    if (prev_img_.empty()) {
        if (!seg_labels.empty()) {
            double mn, mx; cv::minMaxLoc(seg_labels, &mn, &mx);
        }
        result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        result.z_obs = result.z_refined.clone();
        result.variance = cv::Mat(H, W, CV_32F, cv::Scalar(config_.fusion.max_var));
        prev_img_ = img_resized.clone();
        prev_d_rel_ = d_rel_f.clone();
        return result;
    }

    // Baseline guard: skip triangulation when baseline too small/large
    if (handle_baseline_guard(img_resized, d_rel_f, baseline, H, W, result, flow)) {
        return result;
    }

    // ======== [1] Optical Flow + Labels ========
    auto flow_seg = prepare_flow_and_seg(img_resized, seg_labels, flow);


    // Pre-build label index once (reused by apply_metric_scale)
    LabelIndex label_index = build_label_index(flow_seg.seg_labels, flow_seg.num_segments);

    // ======== Iterative Refinement Loop ========
    // Iteration 0: use original inv_depth from monocular model
    // Iteration 1+: use refined depth from previous iteration
    const int total_iters = 1 + config_.iterative;

    cv::Mat z_prior_flow, V_warp_flow;  // Warped prior from prev_depth_ (for iter 0)
    cv::Mat z_prior, V_prior;           // Prior for current iteration (updated each iter)
    const FusionConfig& fuse_cfg = config_.fusion;

    // Declare fusion state variables outside loop (visible after loop ends)
    cv::Mat V_post = cv::Mat(H, W, CV_32F, cv::Scalar(100.0f));
    cv::Mat z_fused;  // Sparse fusion result (before solve_metric_from_rel)

    bool has_prev_depth = !prev_depth_.empty();

    // Pre-compute dense warp from prev_depth_ (for iteration 0)
    if (has_prev_depth) {
        warp_dense_combined(prev_depth_, prev_V_, flow_seg.flow_fwd, cam_,
                           fuse_cfg.lambda_forget, fuse_cfg.min_var,
                           z_prior_flow, V_warp_flow);
    }

    for (int iter = 0; iter < total_iters; ++iter) {
        // For iter > 0: use backward motion estimation with refined depth
        // This resolves forward motion degeneracy by using the refined depth
        // at frame t to constrain the motion estimation
        bool use_backward_estimation = (iter > 0) && !result.z_refined.empty();


        // ======== Prepare flow and depth for this iteration ========
        // Lazy backward flow: compute only when actually needed
        if (use_backward_estimation && flow_seg.flow_bwd.empty()) {
            flow_seg.flow_bwd = compute_flow(img_resized, prev_img_);
        }
        cv::Mat iter_flow = use_backward_estimation ? flow_seg.flow_bwd : flow_seg.flow_fwd;
        cv::Mat iter_d_rel = use_backward_estimation
            ? compute_inv_depth_from_refined(result.z_refined, H, W)
            : prev_d_rel_;

        // Set prior: iter 0 uses warped prev, iter>0 uses previous z_refined
        if (iter == 0) {
            z_prior = z_prior_flow;
            V_prior = V_warp_flow;
        } else {
            z_prior = result.z_refined.clone();
            V_prior = V_post.clone();
        }

        // ======== Motion estimation + pose setup ========
        auto pose_opt = setup_pose(use_backward_estimation, iter_flow, iter_d_rel,
                                    sky_mask, baseline, external_R, external_t);

        if (!pose_opt) {
            // Motion failed
            if (iter == 0) {
                result.z_obs = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
                if (has_prev_depth) {
                    prev_depth_.copyTo(result.z_refined);
                    result.variance = cv::Mat(H, W, CV_32F, cv::Scalar(10.0f));
                    update_state(img_resized, d_rel_f, prev_depth_, prev_V_);
                } else {
                    result.z_refined = cv::Mat(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
                    result.variance = cv::Mat(H, W, CV_32F, cv::Scalar(config_.fusion.max_var));
                    prev_img_ = img_resized.clone();
                    prev_d_rel_ = d_rel_f.clone();
                }
                return result;
            }
            break;  // iter>0: use previous result
        }

        auto& pose = *pose_opt;
        result.pose = pose.pose;


        // Now call compute_tri_and_warp for BOTH GT and normal paths
        // (GT path previously triangulated inline, now unified)
        TriResult tri_result;
        cv::Mat z_prior_pose, V_warp_pose;
        auto tw = compute_tri_and_warp(
            pose.pose_tri, pose.matches,
            has_prev_depth,
            prev_depth_, prev_V_,
            fuse_cfg.lambda_forget, fuse_cfg.min_var
        );
        tri_result = std::move(tw.tri);
        z_prior_pose = std::move(tw.z_prior_pose);
        V_warp_pose = std::move(tw.V_warp_pose);

        result.z_obs = tri_result.z_obs;

        filter_outliers(result.z_obs, d_rel_f, H, W);

        // ========== [3] Bayesian Scale Fusion ==========
        V_post.setTo(100.0f);

        if (!has_prev_depth) {
            // First frame with triangulation: pass through observation
            auto fuse_out = fusion_.first_frame(
                result.z_obs, tri_result.rho, baseline, baseline_state_.b_ref(), fuse_cfg, H, W
            );
            result.z_obs.copyTo(result.z_refined);
            V_post = fuse_out.V_post;
        } else {
            // Select warp prior: pose warp vs flow warp
            auto prior = select_warp_prior(z_prior_pose, V_warp_pose, z_prior_flow, V_prior, H, W);
            cv::Mat z_selected_prior = prior.z_prior;
            V_prior = prior.V_prior;

            // Fuse prior + observation via BayesianFusion 
            auto fuse_out = fusion_.fuse(
                z_selected_prior, V_prior,
                result.z_obs, tri_result.rho, d_rel_f,
                baseline, baseline_state_.b_ref(), fuse_cfg
            );
            result.z_refined = fuse_out.z_refined;
            V_post = fuse_out.V_post;

        }

        // Metric scale estimation
        if (config_.verbose) z_fused = result.z_refined.clone();
        apply_metric_scale(result.z_refined, V_post, rel_depth, sky_mask, label_index, H, W);

        // Save z_fused (verbose only)
        if (config_.verbose && !z_fused.empty()) {
            result.z_fused = z_fused.clone();
        }

    }  // End of iteration loop

    // Post-process: sky → inf, beyond max_depth → inf
    {
        cv::Mat inf_mask = (result.z_refined > config_.max_depth);
        if (config_.outdoor) inf_mask |= (sky_mask == 0);
        result.z_refined.setTo(std::numeric_limits<float>::infinity(), inf_mask);
        V_post.setTo(config_.fusion.max_var, inf_mask);
    }

    result.variance = V_post.clone();

    update_state(img_resized, d_rel_f, result.z_refined, V_post);

    return result;
}

// ============================================================================
// Helpers for refine()
// ============================================================================

Eigen::Vector3d PTCDepth::normalize_t(const Eigen::Vector3d& t, float baseline) {
    return (t.norm() > 1e-8) ? t.normalized() * baseline : Eigen::Vector3d(0, 0, -baseline);
}

std::optional<Eigen::Vector3d> PTCDepth::validate_external_rotation(
    const std::optional<Eigen::Matrix3d>& external_R
) const {
    if (!external_R.has_value()) return std::nullopt;

    const auto& R = external_R.value();
    for (int i = 0; i < 9; ++i)
        if (!std::isfinite(R.data()[i])) return std::nullopt;

    Eigen::Vector3d omega = rotation_matrix_to_omega(R.transpose());  // point→camera frame
    if (!omega.allFinite() || omega.norm() >= 10.0) return std::nullopt;

    return omega;
}

cv::Mat PTCDepth::compute_inv_depth_from_refined(const cv::Mat& z_refined, int H, int W) const {
    cv::Mat z_finite;
    cv::compare(z_refined, z_refined, z_finite, cv::CMP_EQ);
    cv::Mat valid = z_finite & (z_refined > 0.5f) & (z_refined < config_.max_depth);
    cv::Mat inv_depth = cv::Mat::zeros(H, W, CV_32F);
    cv::divide(1.0f, z_refined, inv_depth, 1.0, CV_32F);
    inv_depth.setTo(0.0f, ~valid);
    return inv_depth;
}

PTCDepth::PriorSelection PTCDepth::select_warp_prior(
    const cv::Mat& z_pose, const cv::Mat& V_pose,
    const cv::Mat& z_flow, const cv::Mat& V_flow, int H, int W
) const {
    // Always prefer pose warp
    if (!z_pose.empty()) return {z_pose, V_pose.empty() ? V_flow : V_pose};
    return {z_flow, V_flow};
}

std::optional<PTCDepth::TriangulationPose> PTCDepth::setup_pose(
    bool use_backward,
    const cv::Mat& flow, const cv::Mat& d_rel, const cv::Mat& mask,
    float baseline,
    const std::optional<Eigen::Matrix3d>& external_R,
    const std::optional<Eigen::Vector3d>& external_t
) {
    // If both R and t are provided externally, skip motion estimation entirely
    if (external_R.has_value() && external_t.has_value()) {
        Eigen::Matrix3d R = external_R.value();
        Eigen::Vector3d t = external_t.value();

        const int H = config_.H, W = config_.W;


        Correspondences matches;
        matches.u0.reserve(H * W);
        matches.v0.reserve(H * W);
        matches.u1.reserve(H * W);
        matches.v1.reserve(H * W);

        for (int y = 0; y < H; ++y) {
            const float* flow_row = flow.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                float fx = flow_row[x * 2 + 0];
                float fy = flow_row[x * 2 + 1];
                if (!std::isfinite(fx) || !std::isfinite(fy)) continue;
                matches.u0.push_back(static_cast<float>(x));
                matches.v0.push_back(static_cast<float>(y));
                matches.u1.push_back(static_cast<float>(x) + fx);
                matches.v1.push_back(static_cast<float>(y) + fy);
            }
        }

        TriangulationPose out;
        out.num_matches = static_cast<int>(matches.u0.size());
        out.pose = make_pose(R, t);
        out.pose_tri = out.pose.inverse();
        out.matches = std::move(matches);
        return out;
    }

    // R only: fix rotation, estimate translation via RANSAC
    auto known_omega = validate_external_rotation(external_R);
    MotionResult motion = estimate_motion(flow, d_rel, mask, known_omega);

    if (!motion.success) return std::nullopt;

    Eigen::Matrix3d R = motion.pose.block<3,3>(0,0);
    Eigen::Vector3d t_dir = motion.pose.block<3,1>(0,3);
    Eigen::Vector3d t_scaled = normalize_t(t_dir, baseline);

    TriangulationPose out;
    out.num_matches = motion.num_inliers;

    if (use_backward) {
        out.pose_tri = make_pose(R, t_scaled);
        out.pose = out.pose_tri.inverse();
        out.matches.u0 = motion.matches.u1;  out.matches.v0 = motion.matches.v1;
        out.matches.u1 = motion.matches.u0;  out.matches.v1 = motion.matches.v0;
        out.is_backward = true;
    } else {
        out.pose = make_pose(R, t_scaled);
        out.pose_tri = out.pose.inverse();
        out.matches = std::move(motion.matches);
    }

    return out;
}
}  // namespace ptc_depth
