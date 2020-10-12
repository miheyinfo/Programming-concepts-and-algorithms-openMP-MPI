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
if(WIN32)
    # Find OpenCV, you may need to add "C:\opencv\build\x64\vc15\bin" to your PATH
    set(OpenCV_DIR "C:\\opencv\\build\\x64\\vc15\\lib")
    set(OpenCV_INCLUDE_DIRS "C:\\opencv\\build\\include")
    include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCV_DIR})
endif()
find_package(OpenCV REQUIRED)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

######  MPI #####
find_package(MPI REQUIRED)

target_include_directories(Programming_concepts_and_algorithms_openMP_MPI PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(Programming_concepts_and_algorithms_openMP_MPI PRIVATE ${MPI_CXX_COMPILE_FLAGS})

add_executable(Programming_concepts_and_algorithms_openMP_MPI main.cpp)

# linking
target_link_libraries(Programming_concepts_and_algorithms_openMP_MPI ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${OpenCV_LIBS})