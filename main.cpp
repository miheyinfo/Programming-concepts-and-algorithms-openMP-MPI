#include <iostream>
#include <filesystem>
#include <omp.h>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;


int main() {
//    printf("Number of Processors: %d\n", omp_get_num_procs());
//
//    #pragma omp parallel for num_threads(55)
//    for (int i = 0; i < 400000; ++i) {
//        if(i%10 == 0){
//            printf("Hello from Thread %d, i: %d \n", omp_get_thread_num(), i);
//        }
//
//    }


    // Search Lenna
    auto pathToLenna = std::filesystem::current_path().string().append("/../lenna.png");
    // Read the image file
    Mat sourceImage = imread(pathToLenna);
    // Check for failure
    if (sourceImage.empty())
    {
        printf("Could not open or find the image");
        return -1;
    }

    namedWindow("Original Image", 1);
    imshow("Original Image", sourceImage);

    waitKey();

    return 0;
}
