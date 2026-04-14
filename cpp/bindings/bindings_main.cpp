#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_ptc_depth(py::module_& m);

PYBIND11_MODULE(ptc_depth_cpp, m) {
    m.doc() = "PTC-Depth C++ core library";

    init_ptc_depth(m);
}
