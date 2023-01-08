cmake_minimum_required(VERSION 3.4...3.18)
project(haloflow)

if (MSVC)
    # Windows
    # Set OpenCVConfig.cmake Search Directory
    set( OpenCV_DIR )
    if( NOT CMAKE_CL_64 )
        set( OpenCV_DIR "c:/src/vcpkg/installed/x86-windows/share/opencv" )
    else()
        set( OpenCV_DIR "C:/src/vcpkg/installed/x64-windows/share/opencv" )
    endif()
endif()


find_package(OpenCV REQUIRED )
find_package(Python3 3.8 EXACT COMPONENTS Development Interpreter REQUIRED)
find_package(PythonInterp)
# find_package(pybind11 CONFIG REQUIRED)

INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS} PUBLIC 
)
link_directories( ${OpenCV_LIB_DIR} )
add_subdirectory(pybind11)

set(SOURCES src/header.h src/preprocessing.cpp src/hv_numerical.cpp src/workflows.cpp)
file(GLOB CVNP src/cvnp/*.h src/cvnp/*.cpp)
pybind11_add_module(haloflow MODULE ${SOURCES} ${CVNP} src/main.cpp)
target_link_libraries(haloflow PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(haloflow
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
