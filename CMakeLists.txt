cmake_minimum_required(VERSION 3.17)
project(Programming_concepts_and_algorithms_openMP_MPI)

set(CMAKE_CXX_STANDARD 17)

######  OpenMP #####
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###### OpenCV #####
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

add_executable(Programming_concepts_and_algorithms_openMP_MPI main.cpp)

# linking
target_link_libraries(Programming_concepts_and_algorithms_openMP_MPI stdc++fs ${OpenCV_LIBS})