cmake_minimum_required(VERSION 3.20)

project(onnxruntime_cpp LANGUAGES CXX)

find_package(
    OpenCV
    REQUIRED
    COMPONENTS
    core
    imgcodecs
    imgproc
)

# 生成compile_commands.json 文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 保存 ONNX Runtime SDK 的根目录
set(
    ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime-linux-x64-1.28.0"
    CACHE PATH
    "Path to the ONNX Runtime C++ SDK"
)

# 在真正编译前检查核心头文件是否存在
if(
    NOT EXISTS
    "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h"
)
  message(
      FATAL_ERROR
      "ONNX Runtime header was not found: ${ONNXRUNTIME_ROOT}"
  )
endif()

# 检查需要链接的动态库是否存在
if(
    NOT EXISTS
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
)
    message(
        FATAL_ERROR
        "ONNX Runtime library was not found: ${ONNXRUNTIME_ROOT}"
    )
endif()

# 外部共享库
add_library(
    onnxruntime::onnxruntime
    SHARED
    IMPORTED
)

# 给外部库补充两个重要属性
set_target_properties(
    onnxruntime::onnxruntime
    PROPERTIES

    # 链接阶段使用的动态库文件。
    IMPORTED_LOCATION
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"

    # 编译阶段搜索 ONNX Runtime 头文件的位置
    INTERFACE_INCLUDE_DIRECTORIES
    "${ONNXRUNTIME_ROOT}/include"
)

# 推理核心库由正确性程序和性能程序共同使用。
add_library(
    inference_core
    STATIC
    src/onnx_model.cpp
    src/image_preprocessor.cpp
    src/classification_postprocessor.cpp
)

target_include_directories(
    inference_core
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"

    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(
    inference_core PRIVATE
    cxx_std_17
)

target_compile_options(
    inference_core PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(
    inference_core
    PUBLIC
    onnxruntime::onnxruntime

    PRIVATE
    ${OpenCV_LIBS}
)

# 正确性检查程序：真实图片 -> Top-K预测。
add_executable(
    ort_inspect
    src/main.cpp
)

target_compile_features(ort_inspect PRIVATE cxx_std_17)

target_compile_options(
    ort_inspect PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(
    ort_inspect PRIVATE
    inference_core
)

# 性能程序：预处理一次 -> 反复执行模型推理。
add_executable(
    ort_benchmark
    src/benchmark_main.cpp
    src/inference_benchmark.cpp
)

target_compile_features(ort_benchmark PRIVATE cxx_std_17)

target_compile_options(
    ort_benchmark PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(
    ort_benchmark PRIVATE
    inference_core
)

# 端到端分段计时程序。
add_executable(
    ort_pipeline_benchmark
    src/pipeline_benchmark_main.cpp
    src/pipeline_benchmark.cpp
)

target_compile_features(
    ort_pipeline_benchmark PRIVATE
    cxx_std_17
)

target_compile_options(
    ort_pipeline_benchmark PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(
    ort_pipeline_benchmark PRIVATE
    inference_core
)
