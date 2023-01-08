#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <pybind11/pybind11.h>
#include "cvnp/cvnp.h"
namespace py = pybind11;
using namespace cv;

constexpr uint16_t HEIGHT = 1536;
constexpr uint16_t WIDTH = 2048;
constexpr uint16_t MAX_DEPTH = 16384;
constexpr uint8_t NUM_CHANNELS = 12;


cv::Mat fileToMatrix(std::string filename);
cv::Mat firstPassMatrix( const cv::Mat& image);
uint16_t getMedianValue(int currentRow, int currentColumn, const cv::Mat& image, int ksize);
cv::Mat ToGreyscale( const cv::Mat& image);
cv::Mat toGreyscaleParallel(const cv::Mat& img);
cv::Mat toSubtractedGreyscaleParallel(const cv::Mat& img);
cv::Mat firstPassMatrixParallel( const cv::Mat& image);
uint16_t calculateMedian(std::vector<uint16_t> neighbours);
py::array process_bg_subtraction(pybind11::array& rawImage, bool shared);
py::array process_raw(pybind11::array& rawImage, bool shared);
py::array back_and_forth(pybind11::array& rawImage);
py::array echo(const pybind11::array& rawImage);