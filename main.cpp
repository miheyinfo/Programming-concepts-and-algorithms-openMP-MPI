#include <iostream>
#include <filesystem>
#include <omp.h>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;


int main() {
    const char pathSeparator = std::filesystem::path::preferred_separator;
    // Search Lenna
    auto pathToLenna = std::filesystem::current_path()
            .string()
            .append(&pathSeparator)
            .append("..")
            .append(&pathSeparator)
            .append("lenna.png");
    // Read the image file
    Mat sourceImage = imread(pathToLenna);
    // Check for failure
    if (sourceImage.empty()) {
        printf("Could not open or find the image");
        return -1;
    }

    unsigned long width = sourceImage.cols;
    unsigned long height = sourceImage.rows;

    // create blank grayscale Mat image object
    Mat grayscale(height, width, CV_8U, Scalar(0));

    #pragma omp parallel for collapse(2) default(none) shared(width, height, sourceImage, grayscale)
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {

            auto pixel = sourceImage.at<Vec3b>(i,j);

            unsigned char blue = pixel[0];
            unsigned char red = pixel[1];
            unsigned char green = pixel[2];

            unsigned char greyScaleColor = red * 0.21 + green * 0.72 + blue * 0.07;

            grayscale.at<unsigned char>(i,j) = saturate_cast<unsigned char>(greyScaleColor);
        }
    }

    namedWindow("Grayscale", 1);
    imshow("Grayscale", grayscale);

    namedWindow("Original Image", 1);
    imshow("Original Image", sourceImage);

    waitKey();

    return 0;
}
