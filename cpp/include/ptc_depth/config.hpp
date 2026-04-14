// PTC-Depth pipeline configuration
#pragma once

#include <cmath>

namespace ptc_depth {

// Bayesian fusion hyperparameter
struct FusionConfig {
    // --- Variance estimation ---
    float tau0_deg = 5.0f;           // Reference angle in degrees
    float sigma2_at_tau = 1.0f;      // Variance scale at reference angle
    float min_var = 3e-3f;           // Minimum variance (prevents overconfidence)
    float max_var = 100.0f;          // Maximum variance

    // --- Bayesian fusion ---
    float chi2_hard = 10.828f;       // chi2 quantile at 99% (hard gate) for 1 DOF
    float kappa_min = 0.35f;         // Paper: κ_min — lower bound on Kalman gain (Eq. 10)
    float gate_loosen = 0.9f;        // Gate loosening for low consistency
    float lambda_forget = 0.4f;      // Process noise factor

    // --- Consistency tolerance (paper: σ_e) ---
    float sigma_e_min = 0.01f;       // Min sigma_e (frame-level tolerance floor)
    float sigma_e_max = 0.05f;       // Max sigma_e (frame-level tolerance cap)
    float sigma_e_ema_beta = 0.8f;   // EMA coefficient for sigma_e smoothing

    // --- Frame rejection ---
    bool frame_reject_enable = true;
    int frame_reject_min_valid = 1000;
    float frame_reject_bad_frac = 0.5f;
};

// Motion field estimation (RANSAC + IRLS)
struct MotionFieldConfig {
    // ── 1. Point collection ────────────────────────────────────────────────
    int   grid_cols               = 6;
    int   grid_rows               = 4;
    float min_flow_px             = 0.01f;
    float margin_x_pct            = 0.0f;
    float margin_y_pct            = 0.0f;

    // ── 2. Point subsampling ───────────────────────────────────────────────
    int   max_points              = 2000;
    int   depth_bins              = 6;

    // ── 3. RANSAC ──────────────────────────────────────────────────────────
    int          ransac_max_iters    = 50;
    int          ransac_min_sample   = 6;
    int          ransac_patience_min = 10;
    float        ransac_cond_thresh  = 1e-4f;
    unsigned int seed                = 42;

    // ── 4. Hypothesis scoring ──────────────────────────────────────────────
    float ransac_mad_scale        = 1.5f;
    float ransac_mad_min          = 0.05f;
    float ransac_mad_max          = 0.5f;
    float cos_sim_thresh          = 0.5f;

    // ── 5. IRLS refinement ─────────────────────────────────────────────────
    int   lo_irls_iters           = 5;
    float huber_delta_rel         = 3.5f;

    // ── 6. Inlier classification ───────────────────────────────────────────
    float classify_mad_scale      = 5.0f;
    float classify_angle_mad_scale = 3.0f;
    float classify_angle_min      = 10.0f;
};

// Metric scale estimation
struct MetricScaleConfig {
    float min_pts_ratio = 0.001f;
    float global_trim_k = 4.5f;
};

// Main pipeline configuration
struct PTCDepthConfig {
    // --- Camera intrinsics (must be set by user) ---
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    int H = 0;
    int W = 0;

    // --- User-facing parameters ---
    float max_depth = 80.0f;
    float min_baseline = 0.05f;
    bool outdoor = true;
    bool verbose = false;
    int iterative = 0;

    // --- Tunable parameters ---
    int ransac_max_iters = 50;
    float min_flow_px = 0.1f;
    float margin_x_pct = 0.05f;
    float margin_y_pct = 0.05f;
    int max_points = 500;
    float lambda_forget = 0.1f;
    float kappa_min = 0.25f;
    float tau0_deg = 1.0f;

    // --- Internal sub-configs (not exposed via bindings) ---
    MotionFieldConfig motion;
    MetricScaleConfig metric;
    FusionConfig fusion;
    float baseline_ema_beta = 0.9f;
    int baseline_hist_len = 200;
    float fill_value = NAN;

    // Sync tunable → sub-configs
    void sync() {
        motion.ransac_max_iters = ransac_max_iters;
        motion.min_flow_px = min_flow_px;
        motion.margin_x_pct = margin_x_pct;
        motion.margin_y_pct = margin_y_pct;
        motion.max_points = max_points;
        fusion.lambda_forget = lambda_forget;
        fusion.kappa_min = kappa_min;
        fusion.tau0_deg = tau0_deg;
    }

    PTCDepthConfig() { sync(); }
};
}  // namespace ptc_depth
