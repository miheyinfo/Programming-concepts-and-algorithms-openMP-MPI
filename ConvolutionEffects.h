//
// Created by tool on 27.10.2020.
//

#ifndef PROGRAMMING_CONCEPTS_AND_ALGORITHMS_OPENMP_MPI_CONVOLUTIONEFFECTS_H
#define PROGRAMMING_CONCEPTS_AND_ALGORITHMS_OPENMP_MPI_CONVOLUTIONEFFECTS_H

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <utility>

#include <vector>
using namespace cv;
using namespace std;


enum EffectType {
    GaussianBlur3x3,
    GaussianBlur5x5,
    MotionBlur,
    Emboss
};

class ConvolutionEffects {

private:
    uint _filterWidth;
    uint _filterHeight;
    uint _width;
    uint _height;
    Mat _sourceImage;

    vector<vector<int>> getFilterForEffectType(EffectType effectType);

public:

    ConvolutionEffects(Mat &sourceImage);
    Mat makeConvolutionMagic(EffectType effectType, double factor, double bias);


};


#endif //PROGRAMMING_CONCEPTS_AND_ALGORITHMS_OPENMP_MPI_CONVOLUTIONEFFECTS_H
