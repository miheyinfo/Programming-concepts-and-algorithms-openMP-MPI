//
// Created by tool on 27.10.2020.
//

#include "ConvolutionEffects.h"

ConvolutionEffects::ConvolutionEffects(Mat &sourceImage) {
    _sourceImage = sourceImage.clone();
    _width = _sourceImage.cols;
    _height = _sourceImage.rows;
}

vector<vector<int>> ConvolutionEffects::getFilterForEffectType(EffectType effectType) {
    if (effectType == EffectType::GaussianBlur3x3) {
        _filterHeight = 3;
        _filterWidth = 3;
        return {{1, 2, 1},
                {2, 4, 2},
                {1, 2, 1}};
    } else if (effectType == EffectType::GaussianBlur5x5) {
        _filterHeight = 5;
        _filterWidth = 5;
        return {
                {1, 4,  6,  4,  1},
                {4, 16, 24, 16, 4},
                {6, 24, 36, 24, 6},
                {4, 16, 24, 16, 4},
                {1, 4,  6,  4,  1},
        };
    } else if (effectType == EffectType::MotionBlur) {
        _filterHeight = 9;
        _filterWidth = 9;
        return {
                {1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1},
        };
    } else if (effectType == EffectType::Emboss) {
        _filterHeight = 3;
        _filterWidth = 3;
        return {{-1, -1,  0},
                {-1,  0,  1},
                {0,  1,  1}};
    }


    return {};
}


Mat ConvolutionEffects::makeConvolutionMagic(EffectType effectType, double factor, double bias) {

    Mat sourceImage = _sourceImage.clone();
    Mat filteredImage = _sourceImage.clone();
    uint width = _width;
    uint height = _height;
    vector<vector<int>> filter = getFilterForEffectType(effectType);
    //apply the filter and parallelize among threads
#pragma omp parallel for default(none) collapse(2) shared(factor, bias, width, height, sourceImage, filter, filteredImage)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            double red = 0.0, green = 0.0, blue = 0.0;

            //multiply every value of the filter with corresponding image pixel
            for (int filterY = 0; filterY < _filterHeight; filterY++) {
                for (int filterX = 0; filterX < _filterWidth; filterX++) {
                    int imageX = (x - _filterWidth / 2 + filterX + width) % width;
                    int imageY = (y - _filterHeight / 2 + filterY + height) % height;
                    auto pixel = sourceImage.at<Vec3b>(imageX, imageY);

                    red += pixel[1] * filter[filterY][filterX];
                    green += pixel[2] * filter[filterY][filterX];
                    blue += pixel[0] * filter[filterY][filterX];
                }
            }

            //truncate values smaller than zero and larger than 255
            auto pixel = filteredImage.at<Vec3b>(x, y);
            pixel[1] = min(max(int(factor * red + bias), 0), 255);
            pixel[2] = min(max(int(factor * green + bias), 0), 255);
            pixel[0] = min(max(int(factor * blue + bias), 0), 255);
            filteredImage.at<Vec3b>(x, y) = pixel;
        }
    }

    return filteredImage;
}
