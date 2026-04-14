#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <opencv2/core.hpp>
#include <optional>
#include "ptc_depth/ptc_depth.hpp"
#include "binding_utils.hpp"

namespace py = pybind11;

void init_ptc_depth(py::module_& m) {
    py::class_<ptc_depth::PTCDepthConfig>(m, "PTCDepthConfig")
        .def(py::init<>())
        // Camera intrinsics (must be set by user)
        .def_readwrite("fx", &ptc_depth::PTCDepthConfig::fx)
        .def_readwrite("fy", &ptc_depth::PTCDepthConfig::fy)
        .def_readwrite("cx", &ptc_depth::PTCDepthConfig::cx)
        .def_readwrite("cy", &ptc_depth::PTCDepthConfig::cy)
        .def_readwrite("H", &ptc_depth::PTCDepthConfig::H)
        .def_readwrite("W", &ptc_depth::PTCDepthConfig::W)
        // User-facing parameters
        .def_readwrite("max_depth", &ptc_depth::PTCDepthConfig::max_depth)
        .def_readwrite("min_baseline", &ptc_depth::PTCDepthConfig::min_baseline)
        .def_readwrite("outdoor", &ptc_depth::PTCDepthConfig::outdoor)
        .def_readwrite("verbose", &ptc_depth::PTCDepthConfig::verbose)
        .def_readwrite("iterative", &ptc_depth::PTCDepthConfig::iterative)
        // Tunable parameters
        .def_readwrite("ransac_max_iters", &ptc_depth::PTCDepthConfig::ransac_max_iters)
        .def_readwrite("min_flow_px", &ptc_depth::PTCDepthConfig::min_flow_px)
        .def_readwrite("margin_x_pct", &ptc_depth::PTCDepthConfig::margin_x_pct)
        .def_readwrite("margin_y_pct", &ptc_depth::PTCDepthConfig::margin_y_pct)
        .def_readwrite("max_points", &ptc_depth::PTCDepthConfig::max_points)
        .def_readwrite("lambda_forget", &ptc_depth::PTCDepthConfig::lambda_forget)
        .def_readwrite("kappa_min", &ptc_depth::PTCDepthConfig::kappa_min)
        .def_readwrite("tau0_deg", &ptc_depth::PTCDepthConfig::tau0_deg)
    ;

    py::class_<ptc_depth::PTCDepth>(m, "PTCDepth")
        .def(py::init([](ptc_depth::PTCDepthConfig cfg) {
            cfg.sync();
            return ptc_depth::PTCDepth(cfg);
        }))
        .def("reset", &ptc_depth::PTCDepth::reset)
        .def("refine",
             [](ptc_depth::PTCDepth& self,
                py::array_t<uint8_t> img_np,
                py::array_t<float> d_rel_np,
                float baseline,
                py::array_t<int32_t> seg_labels_np,
                py::array_t<double> external_R_np,
                py::array_t<double> external_t_np,
                py::array_t<float> flow_np) -> py::dict {
                 auto img_info = img_np.request();
                 cv::Mat img;
                 if (img_info.ndim == 2) {
                     img = cv::Mat(img_info.shape[0], img_info.shape[1], CV_8U, img_info.ptr);
                 } else if (img_info.ndim == 3 && img_info.shape[2] == 3) {
                     img = cv::Mat(img_info.shape[0], img_info.shape[1], CV_8UC3, img_info.ptr);
                 } else {
                     throw std::runtime_error("Image must be HxW (grayscale) or HxWx3 (BGR)");
                 }

                 auto d_rel_info = d_rel_np.request();
                 if (d_rel_info.ndim != 2) {
                     throw std::runtime_error("d_rel must be 2D array (HxW)");
                 }
                 cv::Mat d_rel(d_rel_info.shape[0], d_rel_info.shape[1],
                               CV_32F, d_rel_info.ptr);

                 cv::Mat seg_labels;
                 if (seg_labels_np.size() > 0) {
                     auto seg_info = seg_labels_np.request();
                     seg_labels = cv::Mat(seg_info.shape[0], seg_info.shape[1],
                                         CV_32S, seg_info.ptr);
                 }

                 std::optional<Eigen::Matrix3d> external_R = std::nullopt;
                 if (external_R_np.size() > 0) {
                     auto R_info = external_R_np.request();
                     if (R_info.ndim == 2 && R_info.shape[0] == 3 && R_info.shape[1] == 3) {
                         double* R_data = (double*)R_info.ptr;
                         Eigen::Matrix3d R;
                         size_t row_stride = R_info.strides[0] / sizeof(double);
                         size_t col_stride = R_info.strides[1] / sizeof(double);
                         for (int i = 0; i < 3; ++i)
                             for (int j = 0; j < 3; ++j)
                                 R(i, j) = R_data[i * row_stride + j * col_stride];
                         external_R = R;
                     }
                 }

                 std::optional<Eigen::Vector3d> external_t = std::nullopt;
                 if (external_t_np.size() == 3) {
                     auto t_info = external_t_np.request();
                     double* t_data = (double*)t_info.ptr;
                     size_t stride = t_info.strides[0] / sizeof(double);
                     external_t = Eigen::Vector3d(t_data[0], t_data[stride], t_data[2 * stride]);
                 }

                 cv::Mat flow;
                 if (flow_np.size() > 0) {
                     auto flow_info = flow_np.request();
                     if (flow_info.ndim == 3 && flow_info.shape[2] == 2) {
                         flow = cv::Mat(flow_info.shape[0], flow_info.shape[1],
                                        CV_32FC2, flow_info.ptr);
                     }
                 }

                 auto result = self.refine(img, d_rel, baseline, external_R, external_t, seg_labels, flow);

                 int H = result.z_refined.rows;
                 int W = result.z_refined.cols;

                 py::dict output;
                 output["depth"] = mat_to_numpy_f32(result.z_refined, H, W);
                 output["variance"] = mat_to_numpy_f32(result.variance, H, W);

                 if (self.config().verbose) {
                     py::array_t<double> pose_np({4, 4});
                     double* pose_ptr = pose_np.mutable_data();
                     for (int i = 0; i < 4; ++i)
                         for (int j = 0; j < 4; ++j)
                             pose_ptr[i * 4 + j] = result.pose(i, j);
                     output["pose"] = pose_np;

                     if (!result.z_obs.empty())
                         output["z_obs"] = mat_to_numpy_f32(result.z_obs);
                     if (!result.z_fused.empty())
                         output["z_fused"] = mat_to_numpy_f32(result.z_fused);
                 }

                 return output;
             },
             py::arg("img"),
             py::arg("d_rel"),
             py::arg("baseline"),
             py::arg("seg_labels") = py::array_t<int32_t>(),
             py::arg("external_R") = py::array_t<double>(),
             py::arg("external_t") = py::array_t<double>(),
             py::arg("flow") = py::array_t<float>());
}
