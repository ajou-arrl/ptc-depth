/**
 * Motion field estimation with full RANSAC implementation
 */
#include "ptc_depth/motion_field.hpp"
#include "ptc_depth/utils.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>

namespace ptc_depth {

namespace {

// Motion field DOF: omega(3) + t(3)
constexpr int kMotionDof = 6;

// IRLS convergence: stop when parameter update norm is below this threshold
constexpr double kIrlsConvergenceTol = 1e-8;

struct ClassifyThresholds { double residual; double angle; };

// ============================================================================
struct ScoringResult {
    std::vector<bool> inlier_mask;
    double score = 0.0;
    int inlier_count = 0;
};

// Pre-computed rd shared across RANSAC iterations
struct RansacData {
    std::vector<Eigen::Matrix<double, 2, 3>> B_all, A_all;  // Motion Jacobians
    std::vector<Eigen::Vector2d> obs_px;     // Observed flow (pixel units)
    std::vector<double> d_rels;              // Inverse depth
    std::vector<double> flow_mags;           // Flow magnitude (pixel)
    std::vector<int> cell_ids;               // Spatial grid cell
    int max_cells = 0;
    double fx, fy;                           // Focal length (normalized→pixel)

    int size() const { return static_cast<int>(d_rels.size()); }
};

// ============================================================================
// RANSAC Utility Functions
// ============================================================================

/**
 * Count the number of spatial grid cells occupied by at least one inlier.
 */
int count_occupied_cells(const std::vector<bool>& inlier_mask,
                         const std::vector<int>& cell_ids,
                         int max_cells) {
    if (inlier_mask.empty()) return 0;

    std::vector<int> counts(max_cells, 0);
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            counts[cell_ids[i]]++;
        }
    }

    int occupied_cells = 0;
    for (int c : counts) {
        if (c > 0) occupied_cells++;
    }

    return occupied_cells;
}

/**
 * Predict motion field flow and compute relative residuals for all points.
 * pred_px[i] = (B·ω + d_rel·A·t) in pixels
 * rel_residuals[i] = ||obs_px[i] - pred_px[i]|| / flow_magnitudes[i]
 */
void eval_hypothesis(
    const RansacData& rd,
    const Eigen::Vector3d& omega,
    const Eigen::Vector3d& t_dir,
    std::vector<Eigen::Vector2d>& pred_px,
    std::vector<double>& rel_residuals) {

    int n = rd.size();
    pred_px.resize(n);
    rel_residuals.resize(n);
    for (int i = 0; i < n; ++i) {
        double d = rd.d_rels[i];
        Eigen::Vector2d pred;
        pred(0) = rd.B_all[i].row(0).dot(omega) + rd.A_all[i].row(0).dot(t_dir) * d;
        pred(1) = rd.B_all[i].row(1).dot(omega) + rd.A_all[i].row(1).dot(t_dir) * d;
        pred_px[i] = Eigen::Vector2d(pred(0) * rd.fx, pred(1) * rd.fy);
        rel_residuals[i] = (rd.obs_px[i] - pred_px[i]).norm() / rd.flow_mags[i];
    }
}

// Fill linear system rows: predicted_flow = B·ω + d_rel·A·t
void fill_system_row(
    const Eigen::Matrix<double, 2, 3>& B,
    const Eigen::Matrix<double, 2, 3>& A,
    double d_rel, double w,
    const Eigen::Vector2d& rhs,
    bool translation_only,
    Eigen::MatrixXd& M, Eigen::VectorXd& b, int row) {

    if (translation_only) {
        M.block<1, 3>(row,   0) = A.row(0) * d_rel * w;
        M.block<1, 3>(row+1, 0) = A.row(1) * d_rel * w;
    } else {
        M.block<1, 3>(row,   0) = B.row(0) * w;
        M.block<1, 3>(row+1, 0) = B.row(1) * w;
        M.block<1, 3>(row,   3) = A.row(0) * d_rel * w;
        M.block<1, 3>(row+1, 3) = A.row(1) * d_rel * w;
    }
    b(row)   = rhs(0) * w;
    b(row+1) = rhs(1) * w;
}

/**
 * Compute adaptive MAD-based thresholds for residual and angle error.
 */
static ClassifyThresholds mad_thresholds(
    const std::vector<double>& sample_residuals,
    const std::vector<double>& sample_angle_errors,
    const MotionFieldConfig& cfg) {

    ClassifyThresholds thr;
    thr.residual = compute_mad_threshold(sample_residuals, cfg.classify_mad_scale);

    constexpr double kAngleThreshDefault = 45.0;
    constexpr double kAngleThreshMax = 60.0;
    constexpr int kAngleMinSamples = 10;

    thr.angle = kAngleThreshDefault;
    if (static_cast<int>(sample_angle_errors.size()) > kAngleMinSamples) {
        thr.angle = compute_mad_threshold(sample_angle_errors, cfg.classify_angle_mad_scale);
        thr.angle = std::max(static_cast<double>(cfg.classify_angle_min),
                             std::min(thr.angle, kAngleThreshMax));
    }
    return thr;
}

/**
 * Pre-compute motion Jacobians B and A for all points.
 * B (2x3): rotation term; A (2x3): translation term.
 */
static void motion_jacobians(
    const std::vector<Eigen::Vector2d>& coords,
    std::vector<Eigen::Matrix<double, 2, 3>>& B_all,
    std::vector<Eigen::Matrix<double, 2, 3>>& A_all) {

    int n = static_cast<int>(coords.size());
    B_all.resize(n);
    A_all.resize(n);
    for (int i = 0; i < n; ++i) {
        compute_motion_matrices(coords[i](0), coords[i](1), B_all[i], A_all[i]);
    }
}

}  // namespace

namespace motion {

// MAD-based scoring with optional direction check
ScoringResult score_hypothesis(
    const std::vector<double>& rel_residuals,
    const std::vector<Eigen::Vector2d>& pred_px,
    const std::vector<Eigen::Vector2d>& obs_px,
    const std::vector<int>* cell_ids,  // nullptr = skip coverage bonus
    int max_cells,
    const MotionFieldConfig& cfg) {

    int n = static_cast<int>(rel_residuals.size());
    ScoringResult result;
    result.inlier_mask.resize(n, false);
    result.score = 0.0;
    result.inlier_count = 0;

    double thresh = compute_mad_threshold(rel_residuals, cfg.ransac_mad_scale);
    thresh = std::max(static_cast<double>(cfg.ransac_mad_min),
                      std::min(thresh, static_cast<double>(cfg.ransac_mad_max)));

    for (int i = 0; i < n; ++i) {
        bool is_inlier = (rel_residuals[i] <= thresh);

        // Direction check
        if (is_inlier) {
            double pred_mag = pred_px[i].norm() + kEpsDenom;
            double obs_mag = obs_px[i].norm() + kEpsDenom;
            double cos_sim = pred_px[i].dot(obs_px[i]) / (pred_mag * obs_mag);
            if (cos_sim < cfg.cos_sim_thresh) {
                is_inlier = false;
            }
        }

        result.inlier_mask[i] = is_inlier;
        if (is_inlier) {
            result.inlier_count++;
            result.score += 1.0;
        }
    }

    // Cell coverage bonus
    if (cell_ids) {
        result.score += count_occupied_cells(result.inlier_mask, *cell_ids, max_cells);
    }

    return result;
}

// IRLS refinement
Eigen::VectorXd refine_irls(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const RansacData& rd,
    const Eigen::VectorXd& theta_init,
    const MotionFieldConfig& cfg,
    const std::optional<Eigen::Vector3d>& known_omega) {

    int n_points = static_cast<int>(flow_normalized.size());
    bool translation_only = known_omega.has_value();
    int dof = translation_only ? 3 : kMotionDof;

    const auto& B_all = rd.B_all;
    const auto& A_all = rd.A_all;
    const auto& d_rels = rd.d_rels;

    Eigen::VectorXd theta = theta_init;
    double delta = cfg.huber_delta_rel;

    std::vector<Eigen::Vector2d> pred_px(n_points);
    std::vector<double> rel_residuals(n_points);
    std::vector<double> weights(n_points);
    Eigen::MatrixXd M_weighted(2 * n_points, dof);
    Eigen::VectorXd b_weighted(2 * n_points);

    for (int iter = 0; iter < cfg.lo_irls_iters; ++iter) {
        Eigen::Vector3d omega = theta.head<3>();
        Eigen::Vector3d t_dir = theta.tail<3>();

        eval_hypothesis(rd, omega, t_dir, pred_px, rel_residuals);

        for (int i = 0; i < n_points; ++i)
            weights[i] = (rel_residuals[i] <= delta) ? 1.0 : (delta / (rel_residuals[i] + kEpsDenom));

        for (int i = 0; i < n_points; ++i) {
            double d_rel = d_rels[i];
            double w = std::sqrt(weights[i]);
            Eigen::Vector2d rhs = translation_only
                ? flow_normalized[i] - B_all[i] * omega
                : flow_normalized[i];
            fill_system_row(B_all[i], A_all[i], d_rel, w, rhs, translation_only,
                            M_weighted, b_weighted, 2*i);
        }

        Eigen::VectorXd partial = M_weighted.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_weighted);

        Eigen::VectorXd theta_new(kMotionDof);
        if (translation_only)
            theta_new << omega, partial;
        else
            theta_new = partial;

        if ((theta_new - theta).norm() <= kIrlsConvergenceTol * (theta.norm() + kEpsDenom)) {
            theta = theta_new;
            break;
        }
        theta = theta_new;
    }

    return theta;
}

// ============================================================================
// RANSAC estimation of motion field parameters
//
// When known_omega is provided (translation-only mode), omega is fixed and only
// the translational velocity t is estimated (3-DOF). Otherwise, both omega and t
// are estimated jointly (6-DOF). The return value is always a 6D vector [omega; t].
// ============================================================================
Eigen::VectorXd ransac(
    const std::vector<Eigen::Vector2d>& flow_normalized,
    const std::vector<Eigen::Vector2d>& coords_normalized,
    const std::vector<double>& d_rels_in,
    const std::vector<double>& flow_magnitudes_in,
    const std::vector<int>& cell_ids_in,
    int max_cells,
    const MotionFieldConfig& cfg,
    double fx, double fy,
    const std::optional<Eigen::Vector3d>& known_omega) {

    bool translation_only = known_omega.has_value();
    int dof = translation_only ? 3 : kMotionDof;

    int n_points = static_cast<int>(flow_normalized.size());
    if (n_points < cfg.ransac_min_sample) {
        throw std::runtime_error("Not enough points for RANSAC");
    }

    // Build shared RANSAC rd (pre-computed once)
    RansacData rd;
    motion_jacobians(coords_normalized, rd.B_all, rd.A_all);
    rd.d_rels = d_rels_in;
    rd.flow_mags = flow_magnitudes_in;
    rd.cell_ids = cell_ids_in;
    rd.max_cells = max_cells;
    rd.fx = fx;
    rd.fy = fy;
    rd.obs_px.resize(n_points);
    for (int i = 0; i < n_points; ++i)
        rd.obs_px[i] = Eigen::Vector2d(flow_normalized[i](0) * fx, flow_normalized[i](1) * fy);

    // Aliases for readability
    const auto& B_all = rd.B_all;
    const auto& A_all = rd.A_all;
    const auto& d_rels = rd.d_rels;

    // flow_eff: flow with rotation removed (translation-only mode)
    // When not translation-only, alias the original to avoid copying
    std::vector<Eigen::Vector2d> flow_eff_storage;
    if (translation_only) {
        flow_eff_storage.resize(n_points);
        for (int i = 0; i < n_points; ++i)
            flow_eff_storage[i] = flow_normalized[i] - B_all[i] * known_omega.value();
    }
    const auto& flow_eff = translation_only ? flow_eff_storage : flow_normalized;

    // Build cell-to-indices map (dense vector instead of unordered_map)
    int n_cells = rd.max_cells > 0 ? rd.max_cells : (*std::max_element(rd.cell_ids.begin(), rd.cell_ids.end()) + 1);
    std::vector<std::vector<int>> cell_to_idx(n_cells);
    for (int i = 0; i < n_points; ++i) {
        cell_to_idx[rd.cell_ids[i]].push_back(i);
    }

    std::vector<int> unique_cells;
    unique_cells.reserve(n_cells);
    for (int c = 0; c < n_cells; ++c) {
        if (!cell_to_idx[c].empty()) {
            unique_cells.push_back(c);
        }
    }

    if (unique_cells.empty()) {
        throw std::runtime_error("No valid cells for RANSAC");
    }

    Eigen::VectorXd best_theta;
    double best_score = -1.0;
    int no_improve = 0;
    int patience = std::max(cfg.ransac_patience_min, cfg.ransac_max_iters / 10);

    std::mt19937 gen;
    if (cfg.seed == 0) {
        std::random_device rdev;
        gen.seed(rdev());
    } else {
        gen.seed(cfg.seed);
    }

    // Pre-allocate buffers reused across RANSAC iterations
    std::vector<Eigen::Vector2d> pred_px;
    std::vector<double> rel_residuals;
    std::vector<int> sampled_cells;
    std::vector<int> sample_indices;
    sampled_cells.reserve(cfg.ransac_min_sample);
    sample_indices.reserve(cfg.ransac_min_sample);
    const int max_sample_rows = 2 * cfg.ransac_min_sample;
    Eigen::MatrixXd M_sample(max_sample_rows, dof);
    Eigen::VectorXd b_sample(max_sample_rows);

    for (int iter = 0; iter < cfg.ransac_max_iters; ++iter) {
        // Sample cells
        int n_sample_cells = std::min(cfg.ransac_min_sample, static_cast<int>(unique_cells.size()));

        sampled_cells.clear();
        std::sample(unique_cells.begin(), unique_cells.end(),
                    std::back_inserter(sampled_cells), n_sample_cells, gen);


        sample_indices.clear();
        for (int cell : sampled_cells) {
            const auto& indices = cell_to_idx[cell];
            std::uniform_int_distribution<> dis(0, static_cast<int>(indices.size()) - 1);
            sample_indices.push_back(indices[dis(gen)]);
        }

        int n_sample = static_cast<int>(sample_indices.size());

        // Build sample system: M_sample (2n × dof), b_sample from flow_eff
        // Use pre-allocated buffer; only fill the first 2*n_sample rows
        int rows_used = 2 * n_sample;
        for (int j = 0; j < n_sample; ++j) {
            int idx = sample_indices[j];
            fill_system_row(B_all[idx], A_all[idx], d_rels[idx], 1.0,
                            flow_eff[idx], translation_only, M_sample, b_sample, 2*j);
        }

        // Solve via SVD (robust to near-singular systems)
        auto M_block = M_sample.topRows(rows_used);
        auto b_block = b_sample.head(rows_used);
        auto svd = M_block.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Skip degenerate samples via SVD singular value ratio
        auto sv = svd.singularValues();
        if (sv.size() > 0 && sv(sv.size() - 1) / std::max(sv(0), kEpsDenom) < cfg.ransac_cond_thresh) {
            no_improve++;
            if (no_improve >= patience) break;
            continue;
        }

        Eigen::VectorXd partial = svd.solve(b_block);

        Eigen::VectorXd theta_hyp(kMotionDof);
        if (translation_only)
            theta_hyp << known_omega.value(), partial;
        else
            theta_hyp = partial;

        if (!theta_hyp.allFinite()) {
            no_improve++;
            if (no_improve >= patience) break;
            continue;
        }

        // Score: prediction from full theta vs raw obs_px (consistent for both modes)
        Eigen::Vector3d omega_hyp = theta_hyp.head<3>();
        Eigen::Vector3d t_hyp = theta_hyp.tail<3>();

        eval_hypothesis(rd, omega_hyp, t_hyp, pred_px, rel_residuals);

        auto scoring = score_hypothesis(rel_residuals, pred_px, rd.obs_px, &rd.cell_ids, rd.max_cells, cfg);

        if (scoring.score > best_score) {
            best_theta = theta_hyp;
            best_score = scoring.score;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= patience) break;
        }
    }

    if (best_score < 0) {
        throw std::runtime_error("RANSAC failed to find valid solution");
    }

    // IRLS refinement on inliers
    Eigen::Vector3d omega_final = best_theta.head<3>();
    Eigen::Vector3d t_final = best_theta.tail<3>();

    std::vector<Eigen::Vector2d> final_pred_px;
    std::vector<double> final_rel_residuals;
    eval_hypothesis(rd, omega_final, t_final, final_pred_px, final_rel_residuals);

    auto final_scoring = score_hypothesis(final_rel_residuals, final_pred_px, rd.obs_px, nullptr, 0, cfg);

    // Build inlier-only RansacData for IRLS refinement
    std::vector<Eigen::Vector2d> inlier_flow;
    RansacData inlier_rd;
    inlier_rd.fx = rd.fx;
    inlier_rd.fy = rd.fy;
    int n_inliers_est = final_scoring.inlier_count;
    inlier_flow.reserve(n_inliers_est);
    inlier_rd.B_all.reserve(n_inliers_est);
    inlier_rd.A_all.reserve(n_inliers_est);
    inlier_rd.obs_px.reserve(n_inliers_est);
    inlier_rd.d_rels.reserve(n_inliers_est);
    inlier_rd.flow_mags.reserve(n_inliers_est);
    for (int i = 0; i < n_points; ++i) {
        if (final_scoring.inlier_mask[i]) {
            inlier_flow.push_back(flow_normalized[i]);
            inlier_rd.B_all.push_back(B_all[i]);
            inlier_rd.A_all.push_back(A_all[i]);
            inlier_rd.obs_px.push_back(rd.obs_px[i]);
            inlier_rd.d_rels.push_back(rd.d_rels[i]);
            inlier_rd.flow_mags.push_back(rd.flow_mags[i]);
        }
    }

    if (static_cast<int>(inlier_flow.size()) >= dof) {
        best_theta = refine_irls(inlier_flow, inlier_rd, best_theta, cfg, known_omega);
    }

    return best_theta;
}

} // namespace motion

MotionFieldEstimator::MotionFieldEstimator(const MotionFieldConfig& cfg,
                                           const CameraIntrinsics& intrinsics)
    : cfg_(cfg),
      fx_(intrinsics.fx), fy_(intrinsics.fy),
      inv_fx_(1.0 / intrinsics.fx), inv_fy_(1.0 / intrinsics.fy),
      cx_(intrinsics.cx), cy_(intrinsics.cy) {}

// depth_subsample: Depth-stratified subsampling
void MotionFieldEstimator::depth_subsample(CollectedPoints& pts, const MotionFieldConfig& cfg) {
    int n_total = static_cast<int>(pts.coords_normalized.size());
    int n_depth_bins = cfg.depth_bins;
    int max_cells = pts.max_cells;

    // --- Depth-stratified subsampling ---
    // Compute depth percentiles for binning using nth_element (O(n) instead of O(n log n))
    std::vector<double> pct_d_rels = pts.d_rels;
    constexpr float kDepthPctLo = 0.02f;
    constexpr float kDepthPctHi = 0.98f;
    int p2_idx  = std::max(0, static_cast<int>(n_total * kDepthPctLo));
    int p98_idx = std::min(n_total - 1, static_cast<int>(n_total * kDepthPctHi));
    std::nth_element(pct_d_rels.begin(), pct_d_rels.begin() + p2_idx, pct_d_rels.end());
    double d_rel_lo = pct_d_rels[p2_idx];
    std::nth_element(pct_d_rels.begin() + p2_idx, pct_d_rels.begin() + p98_idx, pct_d_rels.end());
    double d_rel_hi = pct_d_rels[p98_idx];

    if (d_rel_hi <= d_rel_lo) {
        d_rel_hi = d_rel_lo + kEpsRange;
    }
    double inv_span = static_cast<double>(n_depth_bins) / (d_rel_hi - d_rel_lo);


    std::vector<int> depth_bin_ids(n_total);
    for (int i = 0; i < n_total; ++i) {
        double d_rel_clipped = std::max(d_rel_lo, std::min(d_rel_hi, pts.d_rels[i]));
        int bin = static_cast<int>((d_rel_clipped - d_rel_lo) * inv_span);
        depth_bin_ids[i] = std::max(0, std::min(n_depth_bins - 1, bin));
    }

    // Create composite bucket key: depth_bin * n_cells + cell_id
    int n_buckets = n_depth_bins * max_cells;
    std::vector<std::vector<int>> bucket_indices(n_buckets);

    for (int i = 0; i < n_total; ++i) {
        int bucket_key = depth_bin_ids[i] * max_cells + pts.cell_ids[i];
        bucket_indices[bucket_key].push_back(i);
    }

    // Count non-empty buckets and their sizes
    std::vector<int> non_empty_buckets;
    std::vector<int> bucket_sizes;
    for (int b = 0; b < n_buckets; ++b) {
        if (!bucket_indices[b].empty()) {
            non_empty_buckets.push_back(b);
            bucket_sizes.push_back(static_cast<int>(bucket_indices[b].size()));
        }
    }

    int n_non_empty = static_cast<int>(non_empty_buckets.size());
    if (n_non_empty == 0) {
        throw std::runtime_error("No valid buckets for stratified sampling");
    }

    // Allocate quota to each bucket proportionally
    std::vector<int> quota(n_non_empty);
    int total_points = 0;
    for (int i = 0; i < n_non_empty; ++i) {
        total_points += bucket_sizes[i];
    }

    // Proportional allocation with remaining distribution
    int remaining = cfg.max_points;
    for (int i = 0; i < n_non_empty; ++i) {
        int alloc = std::max(1, static_cast<int>(
            static_cast<double>(bucket_sizes[i]) / total_points * cfg.max_points));
        alloc = std::min(alloc, bucket_sizes[i]);
        alloc = std::min(alloc, remaining);
        quota[i] = alloc;
        remaining -= alloc;
    }
    // Distribute remaining quota
    while (remaining > 0) {
        bool can_add = false;
        for (int i = 0; i < n_non_empty && remaining > 0; ++i) {
            if (quota[i] < bucket_sizes[i]) {
                quota[i]++;
                remaining--;
                can_add = true;
            }
        }
        if (!can_add) break;
    }

    // Sample from each bucket
    std::vector<int> sampled_indices;
    sampled_indices.reserve(cfg.max_points);

    for (int i = 0; i < n_non_empty; ++i) {
        int bucket_id = non_empty_buckets[i];
        auto& indices = bucket_indices[bucket_id];
        int q = quota[i];

        // Deterministic: take evenly spaced points
        if (q >= static_cast<int>(indices.size())) {
            for (int idx : indices) {
                sampled_indices.push_back(idx);
            }
        } else {
            double step = static_cast<double>(indices.size()) / q;
            for (int j = 0; j < q; ++j) {
                sampled_indices.push_back(indices[static_cast<int>(j * step)]);
            }
        }
    }

    std::vector<Eigen::Vector2d> flow_sampled, coords_sampled;
    std::vector<double> d_rels_sampled;
    std::vector<int> cell_ids_sampled;

    flow_sampled.reserve(sampled_indices.size());
    coords_sampled.reserve(sampled_indices.size());
    d_rels_sampled.reserve(sampled_indices.size());
    cell_ids_sampled.reserve(sampled_indices.size());

    for (int idx : sampled_indices) {
        flow_sampled.push_back(pts.flow_normalized[idx]);
        coords_sampled.push_back(pts.coords_normalized[idx]);
        d_rels_sampled.push_back(pts.d_rels[idx]);
        cell_ids_sampled.push_back(pts.cell_ids[idx]);
    }

    pts.flow_normalized   = std::move(flow_sampled);
    pts.coords_normalized = std::move(coords_sampled);
    pts.d_rels            = std::move(d_rels_sampled);
    pts.cell_ids          = std::move(cell_ids_sampled);
}


// pixel_norms: Flow pixel magnitude computation
void MotionFieldEstimator::pixel_norms(CollectedPoints& pts, double fx, double fy,
                                       float min_flow_px) {
    // --- [4] Flow pixel magnitudes ---
    pts.flow_mags.resize(pts.flow_normalized.size());
    for (size_t i = 0; i < pts.flow_normalized.size(); ++i) {
        pts.flow_mags[i] = std::sqrt(
            std::pow(pts.flow_normalized[i](0) * fx, 2) +
            std::pow(pts.flow_normalized[i](1) * fy, 2)
        );
        pts.flow_mags[i] = std::max(pts.flow_mags[i], static_cast<double>(min_flow_px));
    }
}

// collect_points: Gather and subsample valid points from flow/depth images
MotionFieldEstimator::CollectedPoints MotionFieldEstimator::collect_points(
    const cv::Mat& flow,
    const cv::Mat& d_rel,
    const cv::Mat& mask) {

    int H = flow.rows;
    int W = flow.cols;
    int grid_cols = cfg_.grid_cols;
    int grid_rows = cfg_.grid_rows;
    int max_cells = grid_rows * grid_cols;

    // Margin boundaries
    int margin_x = static_cast<int>(W * cfg_.margin_x_pct);
    int margin_y = static_cast<int>(H * cfg_.margin_y_pct);

    // Build valid pixel mask using OpenCV matrix ops (SIMD-accelerated)
    cv::Mat flow_channels[2];
    cv::split(flow, flow_channels);
    const cv::Mat& flow_x = flow_channels[0];
    const cv::Mat& flow_y = flow_channels[1];

    // d_rel > 0 and finite (NaN > 0 is false, so this handles both)
    cv::Mat valid = (d_rel > 0);

    // flow finite: NaN != NaN, so (flow_x == flow_x) is false for NaN
    cv::Mat fx_finite, fy_finite;
    cv::compare(flow_x, flow_x, fx_finite, cv::CMP_EQ);
    cv::compare(flow_y, flow_y, fy_finite, cv::CMP_EQ);
    valid &= fx_finite;
    valid &= fy_finite;

    // flow magnitude > min_flow
    cv::Mat flow_mag_sq = flow_x.mul(flow_x) + flow_y.mul(flow_y);
    float min_flow_sq = cfg_.min_flow_px * cfg_.min_flow_px;
    valid &= (flow_mag_sq > min_flow_sq);

    // External mask
    if (!mask.empty()) valid &= mask;

    // Apply margin (zero out edges)
    if (margin_y > 0) {
        valid.rowRange(0, margin_y).setTo(0);
        valid.rowRange(H - margin_y, H).setTo(0);
    }
    if (margin_x > 0) {
        valid.colRange(0, margin_x).setTo(0);
        valid.colRange(W - margin_x, W).setTo(0);
    }

    // Extract valid pixel coordinates
    std::vector<cv::Point> valid_coords;
    cv::findNonZero(valid, valid_coords);
    int n_valid = static_cast<int>(valid_coords.size());

    // Collect points from valid coordinates
    const float* d_rel_ptr = d_rel.ptr<float>();
    const cv::Vec2f* flow_ptr = flow.ptr<cv::Vec2f>();

    CollectedPoints pts;
    pts.max_cells = max_cells;
    pts.all_flow_normalized.resize(n_valid);
    pts.all_coords_normalized.resize(n_valid);
    pts.all_d_rels.resize(n_valid);
    pts.cell_ids.resize(n_valid);
    pts.all_matches.u0.resize(n_valid);
    pts.all_matches.v0.resize(n_valid);
    pts.all_matches.u1.resize(n_valid);
    pts.all_matches.v1.resize(n_valid);

    for (int i = 0; i < n_valid; ++i) {
        int x = valid_coords[i].x;
        int y = valid_coords[i].y;
        int idx = y * W + x;
        float d_val = d_rel_ptr[idx];
        const cv::Vec2f& f = flow_ptr[idx];

        pts.all_coords_normalized[i] = Eigen::Vector2d((x - cx_) * inv_fx_, (y - cy_) * inv_fy_);
        pts.all_flow_normalized[i] = Eigen::Vector2d(f[0] * inv_fx_, f[1] * inv_fy_);
        pts.all_d_rels[i] = static_cast<double>(d_val);
        pts.cell_ids[i] = std::min(y * grid_rows / H, grid_rows - 1) * grid_cols
                        + std::min(x * grid_cols / W, grid_cols - 1);
        pts.all_matches.u0[i] = static_cast<float>(x);
        pts.all_matches.v0[i] = static_cast<float>(y);
        pts.all_matches.u1[i] = static_cast<float>(x) + f[0];
        pts.all_matches.v1[i] = static_cast<float>(y) + f[1];
    }

    if (pts.all_coords_normalized.size() < 6) {
        std::ostringstream oss;
        oss << "Not enough valid points for motion estimation: "
            << pts.all_coords_normalized.size() << " points (need >= 6)";
        throw std::runtime_error(oss.str());
    }

    // Subsample if too many points; otherwise use all points directly
    if (pts.all_coords_normalized.size() > static_cast<size_t>(cfg_.max_points)) {
        // Copy needed: depth_subsample will replace these with subsampled versions
        pts.flow_normalized   = pts.all_flow_normalized;
        pts.coords_normalized = pts.all_coords_normalized;
        pts.d_rels            = pts.all_d_rels;
        depth_subsample(pts, cfg_);
    } else {
        // No subsampling: point directly to all_* data

        pts.flow_normalized   = pts.all_flow_normalized;
        pts.coords_normalized = pts.all_coords_normalized;
        pts.d_rels            = pts.all_d_rels;
    }

    // Compute flow magnitudes for RANSAC
    pixel_norms(pts, fx_, fy_, cfg_.min_flow_px);

    return pts;
}


// Per-point residual and angle error computation
void MotionFieldEstimator::residual_at(int i, const CollectedPoints& pts,
                                       const Eigen::Vector3d& omega,
                                       const Eigen::Vector3d& t_cam,
                                       double& out_residual, double& out_angle,
                                       double& out_obs_mag) const {
    double nx = pts.all_coords_normalized[i](0);  // normalized image coord
    double ny = pts.all_coords_normalized[i](1);
    double d_rel = pts.all_d_rels[i];
    double nx2 = nx * nx, ny2 = ny * ny, nxy = nx * ny;

    double t0 = t_cam(0), t1 = t_cam(1), t2 = t_cam(2);
    double omega0 = omega(0), omega1 = omega(1), omega2 = omega(2);

    // A(nx,ny) · t  (translation Jacobian)
    double A_t_0 = -t0 + nx * t2;
    double A_t_1 = -t1 + ny * t2;

    // B(nx,ny) · ω + d_rel · A · t  (motion field prediction)
    double pred_n0 = nxy * omega0 - (1.0 + nx2) * omega1 + ny * omega2 + d_rel * A_t_0;
    double pred_n1 = (1.0 + ny2) * omega0 - nxy * omega1 - nx * omega2 + d_rel * A_t_1;

    double obs_n0 = pts.all_flow_normalized[i](0);
    double obs_n1 = pts.all_flow_normalized[i](1);

    double err0 = pred_n0 - obs_n0;
    double err1 = pred_n1 - obs_n1;
    out_residual = std::sqrt(err0 * err0 + err1 * err1);

    // Pixel space for angle
    double pred_px0 = pred_n0 * fx_, pred_px1 = pred_n1 * fy_;
    double obs_px0 = obs_n0 * fx_, obs_px1 = obs_n1 * fy_;
    out_obs_mag = std::sqrt(obs_px0 * obs_px0 + obs_px1 * obs_px1);

    if (out_obs_mag > cfg_.min_flow_px) {
        double pred_mag = std::sqrt(pred_px0 * pred_px0 + pred_px1 * pred_px1) + kEpsDenom;
        double dot = (pred_px0 * obs_px0 + pred_px1 * obs_px1) / (pred_mag * out_obs_mag);
        dot = std::max(-1.0, std::min(1.0, dot));
        out_angle = std::acos(dot) * 180.0 / M_PI;
    } else {
        out_angle = 0.0;
    }
}

// classify_inliers: Threshold computation + inlier collection from all points
MotionFieldResult MotionFieldEstimator::classify_inliers(
    const Eigen::Vector3d& omega,
    const Eigen::Vector3d& t_cam,
    const CollectedPoints& pts,
    int H, int W) {

    int n_all = static_cast<int>(pts.all_coords_normalized.size());

    // Step 1: Sample points for threshold estimation
    constexpr int kClassifySampleSize = 50000;
    const int threshold_sample_size = std::min(kClassifySampleSize, n_all);
    std::vector<int> sample_indices(n_all);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    if (n_all > threshold_sample_size) {
        int step = n_all / threshold_sample_size;
        sample_indices.clear();
        for (int i = 0; i < n_all; i += step)
            sample_indices.push_back(i);
    }

    // Compute residuals and angle errors on sampled points
    std::vector<double> sample_residuals(sample_indices.size());
    std::vector<double> sample_angle_errors;
    sample_angle_errors.reserve(sample_indices.size());

    for (size_t s = 0; s < sample_indices.size(); ++s) {
        int i = sample_indices[s];
        double residual, angle, obs_mag;
        residual_at(i, pts, omega, t_cam, residual, angle, obs_mag);
        sample_residuals[s] = residual;
        if (obs_mag > cfg_.min_flow_px) {
            sample_angle_errors.push_back(angle);
        }
    }

    // Compute adaptive thresholds (median + k*MAD)
    auto thr = mad_thresholds(sample_residuals, sample_angle_errors, cfg_);
    double residual_thresh = thr.residual;
    double angle_thresh    = thr.angle;

    // Step 2: Collect inliers from all valid pixels
    MotionFieldResult result;
    result.omega = omega;
    double t_cam_norm = t_cam.norm();
    // Normalize translation to unit direction; fall back to forward (+Z) if degenerate (near-zero)
    result.T_hat = (t_cam_norm > 1e-8) ? t_cam / t_cam_norm : Eigen::Vector3d(0, 0, -1);
    result.inlier_mask = cv::Mat::zeros(H, W, CV_8U);

    // Pass 1: compute inlier flags in parallel
    std::vector<uint8_t> is_inlier(n_all, 0);
    int num_inliers = 0;

    #pragma omp parallel for reduction(+:num_inliers) schedule(static)
    for (int i = 0; i < n_all; ++i) {
        double residual, angle, obs_mag;
        residual_at(i, pts, omega, t_cam, residual, angle, obs_mag);

        bool res_ok = (residual <= residual_thresh);
        bool angle_ok = true;
        if (obs_mag > cfg_.min_flow_px) {
            angle_ok = (angle <= angle_thresh);
        }

        if (res_ok && angle_ok) {
            is_inlier[i] = 1;
            num_inliers++;
        }
    }

    // Pass 2: inliers use actual flow, non-inliers use motion field predicted flow
    result.matches.u0.reserve(n_all);
    result.matches.v0.reserve(n_all);
    result.matches.u1.reserve(n_all);
    result.matches.v1.reserve(n_all);

    for (int i = 0; i < n_all; ++i) {
        float u0 = pts.all_matches.u0[i];
        float v0 = pts.all_matches.v0[i];

        if (is_inlier[i]) {
            result.matches.u0.push_back(u0);
            result.matches.v0.push_back(v0);
            result.matches.u1.push_back(pts.all_matches.u1[i]);
            result.matches.v1.push_back(pts.all_matches.v1[i]);

            int px = static_cast<int>(u0);
            int py = static_cast<int>(v0);
            if (px >= 0 && px < W && py >= 0 && py < H) {
                result.inlier_mask.at<uint8_t>(py, px) = 1;
            }
        } else {
            double nx = pts.all_coords_normalized[i](0);
            double ny = pts.all_coords_normalized[i](1);
            double d_rel = pts.all_d_rels[i];
            double nx2 = nx * nx, ny2 = ny * ny, nxy = nx * ny;

            double t0 = t_cam(0), t1 = t_cam(1), t2 = t_cam(2);
            double pred_n0 = nxy*omega(0) - (1.0+nx2)*omega(1) + ny*omega(2) + d_rel*(-t0 + nx*t2);
            double pred_n1 = (1.0+ny2)*omega(0) - nxy*omega(1) - nx*omega(2) + d_rel*(-t1 + ny*t2);

            float pred_u1 = u0 + static_cast<float>(pred_n0 * fx_);
            float pred_v1 = v0 + static_cast<float>(pred_n1 * fy_);

            result.matches.u0.push_back(u0);
            result.matches.v0.push_back(v0);
            result.matches.u1.push_back(pred_u1);
            result.matches.v1.push_back(pred_v1);
        }
    }

    result.num_inliers = num_inliers;

    return result;
}

MotionFieldResult MotionFieldEstimator::estimate(
    const cv::Mat& flow,
    const cv::Mat& d_rel,
    const cv::Mat& mask,
    const std::optional<Eigen::Vector3d>& known_omega) {

    if (flow.type() != CV_32FC2) {
        throw std::runtime_error("Flow must be CV_32FC2");
    }
    if (d_rel.type() != CV_32F) {
        throw std::runtime_error("Inverse depth must be CV_32F");
    }
    if (flow.size() != d_rel.size()) {
        throw std::runtime_error("Flow and inverse depth must have same size");
    }

    // Step 1: Collect and subsample valid points
    CollectedPoints points = collect_points(flow, d_rel, mask);

    // Step 2: RANSAC estimation
    Eigen::VectorXd theta = motion::ransac(
        points.flow_normalized, points.coords_normalized,
        points.d_rels, points.flow_mags, points.cell_ids, points.max_cells,
        cfg_, fx_, fy_, known_omega);
    Eigen::Vector3d omega = theta.head<3>();
    Eigen::Vector3d t_cam = theta.tail<3>();

    // Step 3: Classify all points as inliers/outliers
    return classify_inliers(omega, t_cam, points, flow.rows, flow.cols);
}

} // namespace ptc_depth
