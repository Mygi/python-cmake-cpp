#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "header.h"
#include <stdint.h>
using namespace cv;
using namespace std;

cv::Mat ToGreyscale( const cv::Mat& image) {
  cv::Mat output = cv::Mat(image.rows,image.cols, CV_16UC1);
  for(int i=0; i < image.rows; i++) {
    const cv::Vec3s* rgb = image.ptr<cv::Vec3s>(i);
    uint16_t* Oi = output.ptr<uint16_t>(i);    
    for(int j=0; j < image.cols; j++) {
      int maxValue = rgb[j][0] + rgb[j][1] + rgb[j][2];
      Oi[j] = maxValue / 3;
    }
  }
  return output;
}

cv::Mat toGreyscaleParallel(const cv::Mat& img) {
  cv::Mat output = cv::Mat(img.rows,img.cols, CV_16UC1);
  cv::parallel_for_(cv::Range(0, img.rows * img.cols), [&](const cv::Range& range)
  {    
      for (int r = range.start; r < range.end; r++)
      {       
          int x = r / img.rows;
          int y = r % img.rows;            
          cv::Vec3s rgb = img.at<cv::Vec3s>(y, x);
          int maxValue = rgb[0] + rgb[1] + rgb[2];
          output.ptr<uint16_t>(y)[x] = maxValue / 3;           
          //do work here
      }
  });
  return output;
}
/**
 * Hot pixel removal
 * Likely get stats as well
 * Shift the value to drop invalid values and then
 * make relative to 16bit max
*/
cv::Mat firstPassMatrix( const cv::Mat& image) {
  int indices[4][2] = { {0,0},{0,1},{1,0},{1,1} };
  cv::Mat output = cv::Mat(image.rows,image.cols, CV_16UC1);
  
  for(int index = 0; index < 4; index++) {
    int iStart = indices[index][0];
    int jStart = indices[index][1];
    int currIndex = 0; 
    for(int i = iStart; i < image.rows;i=i+2) {
      const uint16_t* Mi = image.ptr<uint16_t>(i);
      uint16_t* Oi = output.ptr<uint16_t>(i);    
      for(int j=jStart; j <image.cols; j=j+2) {
        const uint16_t current = Mi[j];
        if(current == 0 || current >= MAX_DEPTH) {
           uint16_t updated = getMedianValue(i, j, image, 2);
           uint16_t shifted =  updated >> 4;
           Oi[j] = shifted;
         //cout << "Hot pixel Found with value" << endl;
        } else {
          Oi[j] = current >> 4;
        }      
      }
    }
  }
  return output;
}

cv::Mat firstPassMatrixParallel( const cv::Mat& image) {
  // const int indices[4][2] = { {0,0},{0,1},{1,0},{1,1} };
  cv::Mat output = cv::Mat(image.rows,image.cols, CV_16UC1);  
  //  convert to single iteration - then get median from nearest mosiac channel
  // which means we need to know the current channel 
  // for(int index = 0; index < 4; index++) {
  //   int iStart = indices[index][0];
  //   int jStart = indices[index][1];
    
  cv::parallel_for_(cv::Range(0, image.rows * image.cols), [&](const cv::Range& range)
  {    
     for (int r = range.start; r < range.end; r++)
     {       
          int x = r / image.rows;
          int y = r % image.rows; 
          uint16_t current = image.at<uint16_t>(y,x);
          uint16_t value = current >> 4;
          if(current == 0 || current >= MAX_DEPTH) { 
            value = getMedianValue(y, x, image, 2) >> 4;
          }
       
          output.ptr<uint16_t>(y)[x] = value;
     }
  });
  return output;
}

/**
 * Get the median value from a window
*/
uint16_t getMedianValue(int currentRow, int currentColumn, const cv::Mat& image, const int ksize) {
  std::vector<uint16_t> neighbours;
  for(int i = currentRow - (ksize * 2); i <= currentRow + (ksize * 2); i = i+2) {
    if(i <0 || i >= image.rows){
      continue;
    }
     const uint16_t* Mi = image.ptr<uint16_t>(i);
     for(int j = currentColumn - (ksize * 2); j <= currentColumn + (ksize * 2); j = j+2) {
       if(j<0 || j >= image.cols) {
        continue;
       }
       uint16_t value = Mi[j]; // >= (uint16_t)pow(2,12)
       if(value < MAX_DEPTH && value > 0) {
          neighbours.push_back( value);
       }
    }  
  }
 return calculateMedian(neighbours);
}


cv::Mat fileToMatrix(std::string filename) {
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  is.seekg(0, std::ios::beg);
  char *buffer = new char[HEIGHT*WIDTH*16 + 1];
  if (is.is_open()) 
  {
    is.read(buffer, HEIGHT*WIDTH*16);
    auto imageMatrix = cv::Mat(HEIGHT, WIDTH, CV_16UC1, buffer);
    is.close();
    return imageMatrix;
  }
  return cv::Mat(HEIGHT, WIDTH, CV_16UC1);
}

cv::Mat toSubtractedGreyscaleParallel(const cv::Mat& img) {
    Mat medianBlurred;        
    cv::Mat output = cv::Mat(img.rows, img.cols, CV_16UC1);
    cv::Scalar mean, stddev;
    cv::medianBlur(img, medianBlurred, 5);
        // A histogram would probably tells us about meean and sigma as well
        // 
    cv::meanStdDev(medianBlurred, mean, stddev);
        // This value should be dependent on the density of the image and the size of the particle!
        
    auto noise = mean + 2*stddev; 
    const Mat subtracted = medianBlurred - noise;
    return toGreyscaleParallel(subtracted);
    
}

