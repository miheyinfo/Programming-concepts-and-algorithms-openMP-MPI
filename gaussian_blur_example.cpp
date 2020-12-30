/**
  An example of Gaussian Blur for the "Programmierkonzepte und Algorithmen" course
*/

#include <iostream>
#include <string>

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

// add the math.h library to get the mathematical functions and teh PI value
#define _USE_MATH_DEFINES
#include <cmath>

// include the OpenCV main library
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

const std::string pathSeparator =
#ifdef _WIN32
        "\\";
#else
"/";
#endif

// ---------------------------------------------------------------------------------------------------------------------------

// a type-macro for the Kernel (2D-vector)
typedef std::vector<std::vector<double>> Kernel;

// a function for quickly generating a gaussian kernel
Kernel getGaussianKernel( int kSize, double sigma = 1.0 );

// a C-style macro for getting the pixel [i,j] quickly, returns an array [B, G, R(, A)]
#define PIXEL(image, i, j) ((uchar*)image.data + image.channels() * (static_cast<int>(i) * image.cols + static_cast<int>(j)))

// a function to perform the gaussian blurring
void gaussianBlur( const cv::Mat& inputImage, cv::Mat& outputImage, int kSize, double sigma = 1.0, bool padImage = false);

// function for performing a convolution on one [I,J] pixel in channel K
uchar convolutePixel( const cv::Mat& inputImage, const Kernel& kernel, int I, int J, int K );

// ---------------------------------------------------------------------------------------------------------------------------

// main function
int main( int argc, char** argv )
{
    int rank, size;

    // the full image:
    Mat full_image;

    // image properties:
    int image_properties[4];

    // init MPI
    MPI_Init(&argc, &argv);

    // get the size and rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // load the image ONLY in the master process #0:
    if (rank == 0) {

        // read image
        // Search Lenna
        string pathToLena = ".." + pathSeparator + "lenna.png";
        // Read the image file
        full_image = imread(pathToLena);
        // Check for failure
        if (full_image.empty()) {
            printf("Could not open or find the image");
            return -1;
        }

        // get the properties of the image, to send to other processes later:
        image_properties[0] = full_image.cols; // width
        image_properties[1] = full_image.rows / size; // height, divide it by number of processes
        image_properties[2] = full_image.type(); // image type (in this case: CV_8UC3)
        image_properties[3] = full_image.channels(); // number of channels (here: 3)
    }

    // wait for it to finish:
    MPI_Barrier(MPI_COMM_WORLD);

    // now broadcast the image properties from process #0 to all others:
    // the 'image_properties' array is only initialized in process #0!
    // that's why your IDE might show a warning here
    MPI_Bcast(image_properties, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // now all processes have these properties, initialize the "partial" image in each process
    Mat part_image = Mat(image_properties[1], image_properties[0], image_properties[2]);

    // wait for all to finish:
    MPI_Barrier(MPI_COMM_WORLD);

    // now we can cut the full_image into parts and send the parts to each process!

    // first, the number of bytes to send: (Height * Width * Channels)
    int send_size = image_properties[1] * image_properties[0] * image_properties[3];

    // from process #0 scatter to all others:
    // the Mat.data is a pointer to the raw image data (B1,G1,R1,B2,G2,R2,.....)
    MPI_Scatter(full_image.data, send_size, MPI_UNSIGNED_CHAR, // unsigned char = unsigned 8-bit (byte)
                part_image.data, send_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD); // from process #0

    // of course, you can Bcast the image instead of scattering it

    // now all the PROCESSES have their own copy of the 'part_image' image, which contains
    // a horizontal slice of the image
    // we can do something with it...



  cv::Mat outputImage;
  gaussianBlur( part_image, outputImage, 27, 5.0, true );
  part_image = outputImage;


    imshow("image in process #" + to_string(rank), part_image);

    waitKey(0); // will need to press a key in EACH process...
    destroyAllWindows();

    // save? (each process will save their own image, with different names)
    // imwrite("image_slice_" + to_string(rank) + ".jpg", part_image);

    // you can also, after doing something with the parts, MPI_Gather it back into process #0:

    MPI_Gather(part_image.data, send_size, MPI_UNSIGNED_CHAR,
               full_image.data, send_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Process #0 received the gathered image" << endl;

        imshow("gathered image", full_image);

        waitKey(0); // will need to press a key in EACH process...
        destroyAllWindows();
    }

    // finalize MPI
    MPI_Finalize();

}

// ---------------------------------------------------------------------------------------------------------------------------
/**
  Generates a 2D Gaussian Kernel with the given size (must be an odd value) and sigma
*/
Kernel getGaussianKernelM( int kSize, double sigma )
{
  if ( kSize % 2 == 0 ) {
    throw std::exception( "Kernel size must be an odd number!" );
  }

  Kernel kernel( kSize, std::vector<double>( kSize ) );

  double sum = 0.0;

  for ( int i = 0; i < kSize; i++ ) {
    for ( int j = 0; j < kSize; j++ ) {
      kernel[i][j] = std::exp( -(std::pow( i - kSize / 2, 2 ) + std::pow(j - kSize / 2, 2)) / (2 * std::pow( sigma, 2 )) ) / (2 * 3.14159265359 * std::pow(sigma, 2));
      sum += kernel[i][j];
    }
  }

  for ( int i = 0; i < kSize; i++ ) {
    for ( int j = 0; j < kSize; j++ ) {
      kernel[i][j] /= sum;
    }
  }

  return kernel;
}

/**
  Perform convolution on the [I, J] pixel in the K channel of the input image with the given kernel,
  as a result returns the sum of the multiplied values, e.g. for a Kernel of size [3 x 3]:
  img[I-1,J-1,K] * Kernel[0,0] + img[I-1,J,K] * Kernel[0,1] + ... + img[I+1,J,K] * Kernel[2,1] + img[I+1,J+1,K] * Kernel[2,2]
*/
uchar convolutePixel( const cv::Mat& inputImage, const Kernel& kernel, int I, int J, int K ) {

  int kSize = kernel.size();
  int halfSize = kSize / 2;

  double pixelValue = 0;

  for ( int i = 0; i < kSize; i++ )
  {
    for ( int j = 0; j < kSize; j++ )
    {
      auto pixel = PIXEL( inputImage, I + i - halfSize, J + j - halfSize );
      pixelValue += static_cast<double>(pixel[K]) * kernel[i][j];
    }
  }

  return static_cast<uchar>( std::round( pixelValue ) );
}

/**
  Function to perform a Gaussian Blurring of the input image, with a Gaussian Kernel of given size and sigma
  The result will be written into the outputImage (the image will be overwritten, if outputImage is not empty)
  Setting the 'padImage' flag to TRUE will pad the input image on all sides with zeros, to properly deal
  with the blurring on the image edges
*/
void gaussianBlur( const cv::Mat& inputImage, cv::Mat& outputImage, int kSize, double sigma, bool padImage )
{
  // create a kernel
  Kernel kernel;
  try {
    kernel = getGaussianKernelM( kSize, sigma );
  }
  catch ( std::exception& ex ) {
    std::cerr << "Exception caught when computing a kernel: " << ex.what() << std::endl;
    return;
  }

  // pad image with zeros on all sides (optional step):
  cv::Mat paddedImage;
  if ( padImage ) {
    paddedImage = cv::Mat::zeros( inputImage.rows + kSize - 1, inputImage.cols + kSize - 1, inputImage.type() );
    inputImage.copyTo( paddedImage( cv::Rect( kSize / 2, kSize / 2, inputImage.cols, inputImage.rows ) ) );
  }
  else {
    paddedImage = inputImage.clone();
  }

  // initialize the empty output image:
  outputImage = cv::Mat::zeros( paddedImage.size(), paddedImage.type() );

  // go over the image:
  uchar* pixel;
  for ( int i = kSize / 2; i < paddedImage.rows - kSize / 2; i++ )
  {
    for ( int j = kSize / 2; j < paddedImage.cols - kSize / 2; j++ )
    {
      pixel = PIXEL( outputImage, i, j );
      for ( int k = 0; k < paddedImage.channels(); k++ )
      {
        pixel[k] = convolutePixel( paddedImage, kernel, i, j, k );
      }
    }
  }

  // unpad the image (remove the zero-padding):
  if ( padImage ) {
    outputImage( cv::Rect( kSize / 2, kSize / 2, inputImage.cols, inputImage.rows ) ).copyTo( outputImage );
  }
}