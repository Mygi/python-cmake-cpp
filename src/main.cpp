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


PYBIND11_MODULE(haloflow, m)
{
    m.doc() = "Optimized Halovision Workflows";
    m.def("background_subtraction", &process_bg_subtraction, "Takes a numpy array, returns noise subtracted gresycale image", py::return_value_policy::reference);
    m.def("raw_process", &process_raw, "Takes a numpy array, returns greyscale image", py::return_value_policy::reference);
    m.def("back_and_forth", &back_and_forth, "Takes a numpy array, converts to opnCV matrix and back to numpy", py::return_value_policy::reference);
    m.def("echo", &echo, "Takes a numpy array, converts to opnCV matrix", py::return_value_policy::reference);
}
