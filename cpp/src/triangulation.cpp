#include "ptc_depth/triangulation.hpp"
#include <cmath>
#include <limits>

namespace ptc_depth {

Triangulator::Triangulator(const CameraIntrinsics& cam, const TriangulationConfig& config)
    : cam_(cam), config_(config), K_inv_(cam.K_inv()) {}

static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

// F = K^{-T} [t]_x R K^{-1}
Eigen::Matrix3d compute_fundamental(
    const Eigen::Matrix3d& K_inv,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t
) {
    return K_inv.transpose() * skew(t) * R * K_inv;
}

TriangulationResult Triangulator::triangulate(
    const Correspondences& matches,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t
) {
    TriangulationResult result;
    const int H = cam_.H, W = cam_.W;
    int N = matches.size();

    if (N == 0) {
        result.z1_tri = cv::Mat(H, W, CV_32F, cv::Scalar(config_.fill_value));
        result.rho = cv::Mat(H, W, CV_32F, cv::Scalar(config_.fill_value));
        return result;
    }

    const auto& u0 = matches.u0;
    const auto& v0 = matches.v0;
    const auto& u1 = matches.u1;
    const auto& v1 = matches.v1;

    // r1 = R @ [nx1, ny1, 1]^T (convention: p_curr = R @ p_prev + t)
    const double r00 = R(0, 0), r01 = R(0, 1), r02 = R(0, 2);
    const double r10 = R(1, 0), r11 = R(1, 1), r12 = R(1, 2);
    const double r20 = R(2, 0), r21 = R(2, 1), r22 = R(2, 2);
    double tx = t(0), ty = t(1), tz = t(2);
    float fx_inv = 1.0f / cam_.fx, fy_inv = 1.0f / cam_.fy;

    Eigen::Matrix3d F = compute_fundamental(K_inv_, R, t);
    Eigen::Matrix3d Ft = F.transpose();

    struct ValidPoint {
        int flat_idx;
        float depth;
        float invZ;
        float rho_err;
    };
    std::vector<ValidPoint> valid_points;
    valid_points.reserve(N);

    const double eps = kEpsDenom;

    for (int i = 0; i < N; ++i) {
        double nx0 = (u0[i] - cam_.cx) * fx_inv;
        double ny0 = (v0[i] - cam_.cy) * fy_inv;

        double nx1 = (u1[i] - cam_.cx) * fx_inv;
        double ny1 = (v1[i] - cam_.cy) * fy_inv;
        double r1x = nx1 * r00 + ny1 * r01 + r02;
        double r1y = nx1 * r10 + ny1 * r11 + r12;
        double r1z = nx1 * r20 + ny1 * r21 + r22;

        // Ray-ray intersection: min || r0 * z0 - (r1 * z1 + t) ||^2
        double a = nx0 * nx0 + ny0 * ny0 + 1.0;
        double d = r1x * r1x + r1y * r1y + r1z * r1z;
        double r01_dot = nx0 * r1x + ny0 * r1y + r1z;
        double e0 = nx0 * tx + ny0 * ty + tz;
        double e1 = -(r1x * tx + r1y * ty + r1z * tz);

        double det = a * d - r01_dot * r01_dot;
        if (std::abs(det) <= eps) continue;

        double z0 = (d * e0 + r01_dot * e1) / det;
        double z1 = (r01_dot * e0 + a * e1) / det;

        if (!std::isfinite(z0) || !std::isfinite(z1) || z0 <= 0 || z1 <= 0) continue;

        int ui = static_cast<int>(std::round(u1[i]));
        int vi = static_cast<int>(std::round(v1[i]));
        if (ui < 0 || ui >= cam_.W || vi < 0 || vi >= cam_.H) continue;

        // Back-project to frame 1: P1 = R^T @ (P0 - t)
        double P0x = nx0 * z0, P0y = ny0 * z0, P0z = z0;
        double dx = P0x - tx, dy = P0y - ty, dz = P0z - tz;
        double P1z = R(0, 2) * dx + R(1, 2) * dy + R(2, 2) * dz;

        if (!std::isfinite(P1z) || P1z <= 1e-8) continue;

        // Sampson error (rho)
        Eigen::Vector3d pt0(u0[i], v0[i], 1.0);
        Eigen::Vector3d pt1(u1[i], v1[i], 1.0);
        Eigen::Vector3d Fx0 = F * pt0;
        Eigen::Vector3d Ftx1 = Ft * pt1;
        double d_epipolar = pt1.dot(Fx0);
        double denom = Fx0(0) * Fx0(0) + Fx0(1) * Fx0(1) + Ftx1(0) * Ftx1(0) + Ftx1(1) * Ftx1(1);
        float rho = static_cast<float>((d_epipolar * d_epipolar) / std::max(denom, kEpsDenom));

        int flat = vi * cam_.W + ui;
        valid_points.push_back({flat, static_cast<float>(P1z), 1.0f / static_cast<float>(P1z), rho});
    }

    result.num_valid = static_cast<int>(valid_points.size());

    if (valid_points.empty()) {
        result.z1_tri = cv::Mat(cam_.H, cam_.W, CV_32F, cv::Scalar(config_.fill_value));
        result.rho = cv::Mat(cam_.H, cam_.W, CV_32F, cv::Scalar(config_.fill_value));
        return result;
    }

    // Z-buffer: 2-pass (find closest, then write)
    result.z1_tri = cv::Mat(cam_.H, cam_.W, CV_32F, cv::Scalar(config_.fill_value));
    result.rho = cv::Mat(cam_.H, cam_.W, CV_32F, cv::Scalar(config_.fill_value));

    std::vector<float> max_invZ(cam_.H * cam_.W, -std::numeric_limits<float>::infinity());
    for (const auto& p : valid_points)
        if (p.invZ > max_invZ[p.flat_idx]) max_invZ[p.flat_idx] = p.invZ;

    float* z_ptr = result.z1_tri.ptr<float>(0);
    float* e_ptr = result.rho.ptr<float>(0);
    for (const auto& p : valid_points) {
        if (p.invZ >= max_invZ[p.flat_idx]) {
            z_ptr[p.flat_idx] = p.depth;
            e_ptr[p.flat_idx] = p.rho_err;
        }
    }

    return result;
}
}  // namespace ptc_depth
