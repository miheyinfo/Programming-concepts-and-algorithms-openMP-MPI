#include <iostream>
#include <omp.h>
#include <chrono>
#include <mpi/mpi.h>
#ifdef _WIN32
#include <opencv2/opencv.hpp>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif


using namespace cv;

#define filterWidth 5
#define filterHeight 5

double filter[filterHeight][filterWidth] =
        {
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 1, 1, 0,
                0, 0, 1, 0, 0,
        };

double factor = 1.0 / 13.0;
double bias = 0.0;

int main(int args, char** argv) {

    MPI_Init(&args, &argv);
    MPI_Finalize();

    const std::string pathSeparator =
    #ifdef _WIN32
        "\\";
    #else
        "/";
    #endif

    // Search Lenna
    std::string pathToLena = ".." + pathSeparator + "lenna.png";
    // Read the image file
    Mat sourceImage = imread(pathToLena);
    // Check for failure
    if (sourceImage.empty()) {
        printf("Could not open or find the image");
        return -1;
    }

    unsigned long width = sourceImage.cols;
    unsigned long height = sourceImage.rows;

    // create blank grayscale Mat image object
    Mat grayscale(height, width, CV_8U, Scalar(0));

    auto startTimeGreyScale = std::chrono::high_resolution_clock::now();
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
    auto endTimeGreyScale = std::chrono::high_resolution_clock::now();
    auto durationGreyScale = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeGreyScale - startTimeGreyScale ).count();

    //load the image into the buffer
    Mat filteredImage = sourceImage.clone();

    //apply the filter and parallelize among threads
    auto startTimeGaussianBlur = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for default(none) collapse(2) shared(factor, bias, width, height, sourceImage, filter, filteredImage)
    for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            double red = 0.0, green = 0.0, blue = 0.0;

            //multiply every value of the filter with corresponding image pixel
            for(int filterY = 0; filterY < filterHeight; filterY++)
                for(int filterX = 0; filterX < filterWidth; filterX++)
                {
                    int imageX = (x - filterWidth / 2 + filterX + width) % width;
                    int imageY = (y - filterHeight / 2 + filterY + height) % height;
                    auto pixel = sourceImage.at<Vec3b>(imageX,imageY);
                    red += pixel[1] * filter[filterY][filterX];
                    green += pixel[2] * filter[filterY][filterX];
                    blue += pixel[0] * filter[filterY][filterX];
                }

            //truncate values smaller than zero and larger than 255
            auto pixel = filteredImage.at<Vec3b>(x,y);
            pixel[1] = min(max(int(factor * red + bias), 0), 255);
            pixel[2] = min(max(int(factor * green + bias), 0), 255);
            pixel[0] = min(max(int(factor * blue + bias), 0), 255);
            filteredImage.at<Vec3b>(x,y) = pixel;
        }
    auto endTimeGaussianBlur = std::chrono::high_resolution_clock::now();
    auto durationGaussianBlur = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeGaussianBlur - startTimeGaussianBlur ).count();

    namedWindow("Gaussian Blur 5x5", 1);
    imshow("Gaussian Blur 5x5", filteredImage);

    namedWindow("Grayscale", 1);
    imshow("Grayscale", grayscale);

    namedWindow("Original Image", 1);
    imshow("Original Image", sourceImage);
    std::cout << durationGreyScale << " ms for Grayscale\n";
    std::cout << durationGaussianBlur << " ms for Gaussian Blur";
    waitKey();


    return 0;
}
