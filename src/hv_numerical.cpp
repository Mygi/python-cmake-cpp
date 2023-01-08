#include <stdint.h>
#include "header.h"
using namespace std;

/**
 * Median from a list of values
*/
uint16_t calculateMedian(std::vector<uint16_t> neighbours) {
  if(neighbours.empty()){
    return 0;
  }
  const auto middleItr = neighbours.begin() + neighbours.size() / 2;
  std::nth_element(neighbours.begin(), middleItr, neighbours.end());
  if (neighbours.size() % 2 == 0) {
    const auto leftMiddleItr = std::max_element(neighbours.begin(), middleItr);
    return (*leftMiddleItr + *middleItr) / 2;
  } else {
    return *middleItr;
  }
}