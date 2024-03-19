cmake_minimum_required(VERSION 3.16)
project(atlas_200dk_a2_keypoint)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 11)
add_definitions(-DENABLE_DVPP_INTERFACE)

include_directories(/usr/local/samples/cplusplus/common/acllite/include)
include_directories(/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include)
find_package(OpenCV REQUIRED)
#message(STATUS ${OpenCV_INCLUDE_DIRS})
#添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
#链接Opencv库
add_library(libascendcl SHARED IMPORTED)
set_target_properties(libascendcl PROPERTIES IMPORTED_LOCATION /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libascendcl.so)
add_library(libacllite SHARED IMPORTED Yolov7.cpp Yolov7.h Keypoint.cpp Keypoint.h)
set_target_properties(libacllite PROPERTIES IMPORTED_LOCATION /usr/local/samples/cplusplus/common/acllite/out/aarch64/libacllite.so)


add_executable(atlas_200dk_a2_keypoint main.cpp Yolov7.cpp Cam.cpp Keypoint.cpp)
target_link_libraries(atlas_200dk_a2_keypoint ${OpenCV_LIBS} libascendcl libacllite -ljpeg)