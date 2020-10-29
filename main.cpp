#include <iostream>
#include <omp.h>
#include <chrono>
#ifdef _WIN32
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include "ConvolutionEffects.h"

#else
#include <opencv4/opencv2/opencv.hpp>
#include <mpi/mpi.h>
#include "ConvolutionEffects.h"


#endif
using namespace cv;

using namespace std;

const std::string pathSeparator =
#ifdef _WIN32
        "\\";
#else
"/";
#endif

const int MAXBYTES=8*1024*1024;
uchar buffer[MAXBYTES];


int main(int args, char** argv) {

    int id;
    int count,height,width,type,channels,bytes;
    Mat sourceImage;

    //image properties
    int imageProperties[4];


    ///////----------------
    //  Use this command to run with MPI
    //  Programming-concepts-and-algorithms-openMP-MPI\cmake-build-debug>mpiexec -n 4 Programming_concepts_and_algorithms_openMP_MPI.exe
    //
    //////-----------------
    MPI_Init(&args, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if(id == 0) {

        // read image
        // Search Lenna
        std::string pathToLena = ".." + pathSeparator + "lenna.png";
        // Read the image file
        sourceImage = imread(pathToLena);
        // Check for failure
        if (sourceImage.empty()) {
            printf("Could not open or find the image");
            return -1;
        }

        type  = sourceImage.type();
        channels = sourceImage.channels();
        memcpy(&buffer[0 * sizeof(int)],(uchar*)&sourceImage.rows,sizeof(int));
        memcpy(&buffer[1 * sizeof(int)],(uchar*)&sourceImage.cols,sizeof(int));
        memcpy(&buffer[2 * sizeof(int)],(uchar*)&type,sizeof(int));

        bytes=sourceImage.rows*sourceImage.cols*channels;
        memcpy(&buffer[3*sizeof(int)],sourceImage.data,bytes);

    }
    //wait for image read and broadcast
    MPI_Barrier(MPI_COMM_WORLD);

    // broadcast image size
    MPI_Bcast(&bytes,1,MPI_INT,0,MPI_COMM_WORLD);

    //wait for image size and broadcast
    MPI_Barrier(MPI_COMM_WORLD);

    // broadcast image object
    MPI_Bcast(buffer,bytes,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    memcpy((uchar*)&height,&buffer[0 * sizeof(int)], sizeof(int));
    memcpy((uchar*)&width,&buffer[1 * sizeof(int)], sizeof(int));
    memcpy((uchar*)&type,&buffer[2 * sizeof(int)], sizeof(int));

    sourceImage = Mat(height,width,type,(uchar*)&buffer[3*sizeof(int)]);

    if(id == 0) {
        // show result
        namedWindow("Original Image", 1);
        imshow("Original Image", sourceImage);
    }

    if(id == 1) {
        // create blank grayscale Mat image object
        Mat grayscaleImage = Mat(height, width, CV_8U, Scalar(0));

        auto startTimeGreyScale = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2) default(none) shared(width, height, sourceImage, grayscaleImage)
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {

                auto pixel = sourceImage.at<Vec3b>(i,j);

                unsigned char blue = pixel[0];
                unsigned char red = pixel[1];
                unsigned char green = pixel[2];

                unsigned char greyScaleColor = red * 0.21 + green * 0.72 + blue * 0.07;

                grayscaleImage.at<unsigned char>(i,j) = saturate_cast<unsigned char>(greyScaleColor);
            }
        }
        // show result
        namedWindow("Grayscale", 1);
        imshow("Grayscale", grayscaleImage);

    }
    if (id == 2) {
        ConvolutionEffects convolutionEffects(sourceImage);
        Mat filteredImage = convolutionEffects.makeConvolutionMagic(EffectType::Emboss,1,128);
        // show result
        namedWindow("Emboss", 1);
        imshow("Emboss", filteredImage);
    }
    if (id == 3) {
        ConvolutionEffects convolutionEffects(sourceImage);
        Mat filteredImage = convolutionEffects.makeConvolutionMagic(EffectType::MotionBlur,1/9.0,0);
        // show result
        namedWindow("Motion", 1);
        imshow("Motion", filteredImage);
    }
    if (id == 4) {
        ConvolutionEffects convolutionEffects(sourceImage);
        Mat filteredImage = convolutionEffects.makeConvolutionMagic(EffectType::Sharpen,1.0,0);
        // show result
        namedWindow("Sharpen - Draw Effect", 1);
        imshow("Sharpen - Draw Effect", filteredImage);
    }

    MPI_Finalize();

    waitKey();
    return 0;
}
