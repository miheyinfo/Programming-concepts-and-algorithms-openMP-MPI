#include <iostream>
#include <omp.h>
#include <chrono>
#ifdef _WIN32
#include <opencv2/opencv.hpp>
#include <mpi.h>
#else
#include <opencv4/opencv2/opencv.hpp>
#include <mpi/mpi.h>
#endif


using namespace cv;
using namespace std;

const std::string pathSeparator =
#ifdef _WIN32
        "\\";
#else
"/";
#endif

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


const int MAXBYTES=8*1024*1024;
uchar buffer[MAXBYTES];


int main(int args, char** argv) {

    int id;
    int count,height,width,type,channels,bytes;
    Mat sourceImage;

    //image properties
    int imageProperties[4];

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
        //load the image into the buffer
        Mat filteredImage = sourceImage.clone();

        //apply the filter and parallelize among threads
        auto startTimeGaussianBlur = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for default(none) collapse(2) shared(factor, bias, width, height, sourceImage, filter, filteredImage)
        for(int x = 0; x < width; x++) {
            for(int y = 0; y < height; y++) {
                double red = 0.0, green = 0.0, blue = 0.0;

                //multiply every value of the filter with corresponding image pixel
                for(int filterY = 0; filterY < filterHeight; filterY++) {
                    for(int filterX = 0; filterX < filterWidth; filterX++) {
                        int imageX = (x - filterWidth / 2 + filterX + width) % width;
                        int imageY = (y - filterHeight / 2 + filterY + height) % height;
                        auto pixel = sourceImage.at<Vec3b>(imageX,imageY);
                        red += pixel[1] * filter[filterY][filterX];
                        green += pixel[2] * filter[filterY][filterX];
                        blue += pixel[0] * filter[filterY][filterX];
                    }
                }

                //truncate values smaller than zero and larger than 255
                auto pixel = filteredImage.at<Vec3b>(x,y);
                pixel[1] = min(max(int(factor * red + bias), 0), 255);
                pixel[2] = min(max(int(factor * green + bias), 0), 255);
                pixel[0] = min(max(int(factor * blue + bias), 0), 255);
                filteredImage.at<Vec3b>(x,y) = pixel;
            }
        }
        // show result
        namedWindow("Gaussian Blur 5x5", 1);
        imshow("Gaussian Blur 5x5", filteredImage);
    }

    MPI_Finalize();

    waitKey();
    return 0;
}
