#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pybind11/pybind11.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#include <iostream>
#include <string>
#include <sstream>
#include "cvnp/cvnp.h"
#include "header.h"
using namespace cv;
using namespace std;
namespace py = pybind11;

void pydef_cvnp(pybind11::module& m);

PYBIND11_MODULE(workflows, m)
{
    m.doc() = "Optimized Halovision Workflows";
    m.def("background_subtraction", &process_bg_subtraction, "Takes a numpy array, returns noise subtracted gresycale image", py::return_value_policy::reference);
    m.def("raw_process", &process_raw, "Takes a numpy array, returns greyscale image", py::return_value_policy::reference);
    m.def("back_and_forth", &back_and_forth, "Takes a numpy array, converts to opnCV matrix and back to numpy", py::return_value_policy::reference);
    m.def("echo", &echo, "Takes a numpy array, converts to opnCV matrix", py::return_value_policy::reference);
    pydef_cvnp(m);
  #ifdef VERSION_INFO
      m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
      m.attr("__version__") = "dev";
  #endif
}

int main(int argc, char** argv)
{

  unsigned long long summed =0;
  double sum1 = 0;
  double sum2 = 0;
  double sum3 = 0;
  int numruns = 5;

  Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
 
  for(int run = 0; run < numruns; run++) {
    for(int i =1; i <= NUM_CHANNELS; i++) {
      std::ostringstream oss;
      std::ostringstream outfile;
      outfile << "/home/melissa/tinybrightthings/workspace/HaloVision/hv_share_c/greyscale-" << i << ".jpg"; 
      oss << "/home/melissa/tinybrightthings/workspace/HaloVision/tests/test-files/" << std::setfill('0') << std::setw(4) << i << ".raw";
      std::string filename = oss.str();
      const auto cvArray = fileToMatrix(filename);

        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

        const auto cvArrayMaskedPixels = firstPassMatrixParallel(cvArray);
        std::chrono::time_point<std::chrono::steady_clock> point1 = std::chrono::steady_clock::now();
        auto displayArray =  cv::Mat(HEIGHT, WIDTH, CV_16UC1);
        
        cv::demosaicing(cvArrayMaskedPixels, displayArray, cv::COLOR_BayerGR2BGR);
        std::chrono::time_point<std::chrono::steady_clock> point2 = std::chrono::steady_clock::now();
        
        // Mat medianBlurred;
        
        // cv::Scalar mean, stddev;
        // cv::medianBlur(displayArray, medianBlurred, 5);
        // // A histogram would probably tells us about meean and sigma as well
        // // 
        // cv::meanStdDev(medianBlurred, mean, stddev);
        // // This value should be dependent on the density of the image and the size of the particle!
        
        // auto noise = mean + 2*stddev; 
        // Mat subtracted = medianBlurred - noise;
        const auto greyscale = toSubtractedGreyscaleParallel(displayArray);
       
      
        // Mat fgMask;
        // pBackSub->apply(medianBlurred, fgMask);

        //cv:calcHist()
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::micro> fp_ms = end - start; 
        std::chrono::duration<double, std::micro> fp_ms_1 = point1 - start;
        std::chrono::duration<double, std::micro> fp_ms_2 = point2 - point1;
        std::chrono::duration<double, std::micro> fp_ms_3 = end - point2;

        std::chrono::duration<unsigned long long, std::micro> int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // cout << int_ms.count()  << endl;
        // cout << fp_ms.count()  << endl;
        // cout << fp_ms_1.count()  << endl;
        // cout << fp_ms_2.count()  << endl;
        // cout << fp_ms_3.count()  << endl;
        // auto item1 = displayArray.at<cv::Vec3b>(0,0);

        //greyscale array
        cv::imwrite(outfile.str(),greyscale);
        summed += int_ms.count();
        sum1 += fp_ms_1.count();
        sum2 += fp_ms_2.count();
        sum3 += fp_ms_3.count();

    }
  }
  cout << "Average perfomance" << endl;
  cout << summed / NUM_CHANNELS / 5  << endl; // 176750
  cout << sum1 / NUM_CHANNELS / 5 << endl; //165770
  cout << sum2 / NUM_CHANNELS / 5 << endl; //2242
  cout << sum3 / NUM_CHANNELS / 5  << endl; // 9267
  // // cv::imwrite("/home/melissa/tinybrightthings/workspace/HaloVision/hv_share_c/original.png",cvArray);
  // if (displayArray.empty()) // Check for failure
  // {
  //  cout << "Could not open or find the image" << endl;
  //  system("pause"); //wait for any key press
  //  return -1;
  // }

  // String windowName = "My HelloWorld Window"; //Name of the window

  // namedWindow(windowName); // Create a window

  // imshow(windowName, greyscale); // Show our image inside the created window.

  // waitKey(0); // Wait for any keystroke in the window

//   destroyWindow(windowName); //destroy the created window

  return 0;
}

