cmake_minimum_required(VERSION 3.15)
project(ptc_depth_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RPATH: embed linked library directories (OpenCV etc.) into installed .so
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Optimization flags (portable — no -march=native or -ffast-math)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Dependencies
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc video)
find_package(Eigen3 3.3 REQUIRED)

# Python and pybind11 for bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake, trying pip...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_result
    )
    if(pybind11_result EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Install via: pip install pybind11")
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(ptc_depth_core STATIC
    src/motion_field.cpp
    src/triangulation.cpp
    src/scale_estimation.cpp
    src/depth_warp.cpp
    src/bayesian_fusion.cpp
    src/ptc_depth.cpp
)

target_link_libraries(ptc_depth_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

set_target_properties(ptc_depth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(ptc_depth_cpp
    bindings/bindings_main.cpp
    bindings/ptc_depth_bindings.cpp
)

target_link_libraries(ptc_depth_cpp PRIVATE
    ptc_depth_core
    ${OpenCV_LIBS}
)

install(TARGETS ptc_depth_cpp DESTINATION .)
