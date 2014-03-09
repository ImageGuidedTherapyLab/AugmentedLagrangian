CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(nfft2d)

# Find GADGETRON.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} $ENV{GADGETRON_HOME}/cmake)
FIND_PACKAGE(Gadgetron REQUIRED)
include_directories(BEFORE ${GADGETRON_INCLUDE_DIR})
link_directories(${GADGETRON_LIB_DIR})

# Build Executables
add_executable(nfft main_nfft.cpp)
add_executable(nffth main_nffth.cpp)
add_executable(nffth_cg main_cg.cpp)
add_executable(nffth_sb main_sb.cpp)

target_link_libraries(nfft                gpuoperators gpucore gpunfft hostutils ${CUDA_LIBRARIES})
target_link_libraries(nffth               gpuoperators gpucore gpunfft hostutils ${CUDA_LIBRARIES})
target_link_libraries(nffth_cg gpusolvers gpuoperators gpucore gpunfft hostutils ${CUDA_LIBRARIES})
target_link_libraries(nffth_sb gpusolvers gpuoperators gpucore gpunfft hostutils ${CUDA_LIBRARIES})
