cmake_minimum_required (VERSION 3.13)
project(TensorRT_sampleMNISTAPI LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

find_package (OpenCV REQUIRED)

set (Torch_DIR ${CMAKE_SOURCE_DIR}/libtorch_v1.10.2)
find_package(Torch PATHS ${Torch_DIR}/share/cmake/Torch REQUIRED)

add_executable (main
  main.cpp
)
target_include_directories (main PRIVATE
  common
  /usr/local/cuda-11.4/targets/x86_64-linux/include
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries (main
  nvparsers
  nvinfer
  ${TORCH_LIBRARIES}
  opencv_core
  opencv_imgcodecs
  opencv_imgproc
)
