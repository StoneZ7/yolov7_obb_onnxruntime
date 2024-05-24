cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4 REQUIRED)
set(ONNXRUNTIME_DIR "/home/zl/Downloads/onnxruntime-linux-x64-1.16.3")

include_directories("${ONNXRUNTIME_DIR}/include")

##下面是输出信息
message(STATUS "Opnecv ;ibrary status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")


include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/eigen3) 


add_executable(rotated_yolov7_ort rotated_yolov7_ort.cpp )
target_link_libraries(rotated_yolov7_ort "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ${OpenCV_LIBS} )
target_link_libraries(rotated_yolov7_ort ${OpenCV_LIBS})

