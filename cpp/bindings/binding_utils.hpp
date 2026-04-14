#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// numpy <-> cv::Mat conversion helpers
// ---------------------------------------------------------------------------

// Convert a uint8 numpy array (HxW grayscale or HxWx3 BGR) to cv::Mat.
// Returns a zero-copy view — the numpy array must outlive the returned Mat.
inline cv::Mat numpy_to_mat(py::array_t<uint8_t> arr) {
    py::buffer_info buf = arr.request();

    if (buf.ndim == 2) {
        // Grayscale HxW
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8U, buf.ptr);
    } else if (buf.ndim == 3 && buf.shape[2] == 3) {
        // Color HxWx3
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    } else {
        throw std::runtime_error("Unsupported array shape for image");
    }
}

// Convert a CV_32FC2 flow Mat to an HxWx2 float32 numpy array (zero-copy view).
inline py::array_t<float> mat_to_numpy(const cv::Mat& mat) {
    if (mat.type() != CV_32FC2) {
        throw std::runtime_error("Only CV_32FC2 flow supported");
    }

    return py::array_t<float>(
        {mat.rows, mat.cols, 2},                    // shape
        {mat.step[0], mat.step[1], sizeof(float)},  // strides
        (float*)mat.data                             // data pointer
    );
}

// Convert a CV_32F Mat to an HxW float32 numpy array (copying data).
// If mat is empty, returns a zero-filled array of the given (H, W).
// Pass H=0, W=0 to return an empty array when mat is empty.
inline py::array_t<float> mat_to_numpy_f32(const cv::Mat& mat,
                                            int H = -1, int W = -1) {
    if (mat.empty()) {
        if (H <= 0 || W <= 0) {
            return py::array_t<float>();
        }
        py::array_t<float> arr({H, W});
        std::fill(arr.mutable_data(), arr.mutable_data() + H * W, 0.0f);
        return arr;
    }
    int h = (H > 0) ? H : mat.rows;
    int w = (W > 0) ? W : mat.cols;
    py::array_t<float> arr({h, w});
    float* ptr = arr.mutable_data();
    for (int r = 0; r < h; ++r) {
        std::memcpy(ptr + r * w, mat.ptr<float>(r), w * sizeof(float));
    }
    return arr;
}

// Convert a CV_32S (int32) Mat to an HxW int32 numpy array (copying data).
inline py::array_t<int32_t> mat_to_numpy_i32(const cv::Mat& mat,
                                              int H = -1, int W = -1) {
    if (mat.empty()) {
        if (H <= 0 || W <= 0) return py::array_t<int32_t>();
        py::array_t<int32_t> arr({H, W});
        std::fill(arr.mutable_data(), arr.mutable_data() + H * W, int32_t(0));
        return arr;
    }
    int h = (H > 0) ? H : mat.rows;
    int w = (W > 0) ? W : mat.cols;
    py::array_t<int32_t> arr({h, w});
    int32_t* ptr = arr.mutable_data();
    for (int r = 0; r < h; ++r) {
        std::memcpy(ptr + r * w, mat.ptr<int32_t>(r), w * sizeof(int32_t));
    }
    return arr;
}
