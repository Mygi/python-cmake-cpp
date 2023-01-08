#include <pybind11/pybind11.h>
#include "cvnp/cvnp.h"
#include "header.h"
namespace py = pybind11;

 py::array echo(const pybind11::array& rawImage) {
    return rawImage;
 }
 py::array back_and_forth(pybind11::array& rawImage) {
    Mat matarry = cvnp::nparray_to_mat(rawImage);
    py::array out = cvnp::mat_to_nparray(matarry, false);
    return out;
 }
 py::array process_raw(pybind11::array& rawImage, bool shared) {
    const Mat cvArray = cvnp::nparray_to_mat(rawImage);
    const auto cvArrayMaskedPixels = firstPassMatrixParallel(cvArray);
    auto displayArray =  cv::Mat(cvArrayMaskedPixels.rows, cvArrayMaskedPixels.cols, CV_16UC1);        
    cv::demosaicing(cvArrayMaskedPixels, displayArray, cv::COLOR_BayerGR2BGR);
    const auto greyscale = toGreyscaleParallel(displayArray);
    return cvnp::mat_to_nparray(greyscale, shared);

 } 

 py::array process_bg_subtraction(pybind11::array& rawImage, bool shared) {
    const Mat cvArray = cvnp::nparray_to_mat(rawImage);
    const auto cvArrayMaskedPixels = firstPassMatrixParallel(cvArray);
    auto displayArray =  cv::Mat(cvArrayMaskedPixels.rows, cvArrayMaskedPixels.cols, CV_16UC1);        
    cv::demosaicing(cvArrayMaskedPixels, displayArray, cv::COLOR_BayerGR2BGR);
    const auto greyscale = toSubtractedGreyscaleParallel(displayArray);
    return cvnp::mat_to_nparray(greyscale, shared);
 } 