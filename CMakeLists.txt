cmake_minimum_required(VERSION 3.20)

project(MiniMind LANGUAGES C CXX)

option(MINIMIND_BUILD_TESTS "Build MiniMind tests" ON)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TRAIN ON CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
set(MNN_EVALUATION OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_PROTOBUFFER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_OPENCV OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_AUDIO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_LLM OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DIFFUSION OFF CACHE BOOL "" FORCE)
set(MNN_METAL OFF CACHE BOOL "" FORCE)
set(MNN_OPENCL OFF CACHE BOOL "" FORCE)
set(MNN_OPENGL OFF CACHE BOOL "" FORCE)
set(MNN_VULKAN OFF CACHE BOOL "" FORCE)
set(MNN_CUDA OFF CACHE BOOL "" FORCE)
set(MNN_MUSA OFF CACHE BOOL "" FORCE)
set(MNN_TENSORRT OFF CACHE BOOL "" FORCE)
set(MNN_COREML OFF CACHE BOOL "" FORCE)
set(MNN_NNAPI OFF CACHE BOOL "" FORCE)
set(MNN_QNN OFF CACHE BOOL "" FORCE)
set(MNN_ONEDNN OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_MINI OFF CACHE BOOL "" FORCE)
set(MNN_SKIPBUILD_GEOMETRY OFF CACHE BOOL "" FORCE)
set(MNN_REDUCE_SIZE OFF CACHE BOOL "" FORCE)
set(MNN_USE_THREAD_POOL ON CACHE BOOL "" FORCE)
set(MNN_OPENMP OFF CACHE BOOL "" FORCE)
set(MNN_AVX2 ON CACHE BOOL "" FORCE)
set(MNN_AVX512 OFF CACHE BOOL "" FORCE)
set(MNN_KLEIDIAI OFF CACHE BOOL "" FORCE)

set(mnn_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/3rd_party/MNN")
add_subdirectory("${mnn_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mnn")

# Keep the vendored training archive limited to autograd and optimizers.
get_target_property(_mnn_archive_sources MNN SOURCES)
list(FILTER _mnn_archive_sources EXCLUDE REGEX "MNNTrainUtils")
set_property(TARGET MNN PROPERTY SOURCES "${_mnn_archive_sources}")
get_target_property(_mnn_train_sources MNNTrain SOURCES)
list(FILTER _mnn_train_sources INCLUDE REGEX "[/\\]source[/\\](grad|optimizer)[/\\]")
set_property(TARGET MNNTrain PROPERTY SOURCES "${_mnn_train_sources}")

add_library(
    MiniMind STATIC
    src/loss.cpp
    src/attention.cpp
    src/gru.cpp
    src/state_pooling.cpp
    src/embedding.cpp
)
target_compile_features(MiniMind PUBLIC cxx_std_17)
target_include_directories(
    MiniMind
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${mnn_SOURCE_DIR}/include
        ${mnn_SOURCE_DIR}/schema/current
        ${mnn_SOURCE_DIR}/tools/train/source/grad
)
target_include_directories(MiniMind PRIVATE ${mnn_SOURCE_DIR}/3rd_party/flatbuffers/include)
target_link_libraries(MiniMind PUBLIC MNN)

if(MINIMIND_BUILD_TESTS)
    enable_testing()
    add_executable(minimind_loss_test tests/loss_test.cpp)
    target_include_directories(
        minimind_loss_test
        PRIVATE ${mnn_SOURCE_DIR}/3rd_party/flatbuffers/include
    )
    target_link_libraries(minimind_loss_test PRIVATE MiniMind)
    add_test(NAME minimind.loss COMMAND minimind_loss_test)

    add_executable(minimind_attention_test tests/attention_test.cpp)
    target_link_libraries(minimind_attention_test PRIVATE MiniMind)
    add_test(NAME minimind.attention COMMAND minimind_attention_test)

    add_executable(minimind_gru_test tests/gru_test.cpp)
    target_include_directories(
        minimind_gru_test
        PRIVATE ${mnn_SOURCE_DIR}/3rd_party/flatbuffers/include
    )
    target_link_libraries(minimind_gru_test PRIVATE MiniMind)
    add_test(NAME minimind.gru COMMAND minimind_gru_test)

    add_executable(minimind_state_pooling_test tests/state_pooling_test.cpp)
    target_include_directories(
        minimind_state_pooling_test
        PRIVATE ${mnn_SOURCE_DIR}/3rd_party/flatbuffers/include
    )
    target_link_libraries(minimind_state_pooling_test PRIVATE MiniMind)
    add_test(NAME minimind.state_pooling COMMAND minimind_state_pooling_test)

    add_executable(minimind_embedding_test tests/embedding_test.cpp)
    target_include_directories(
        minimind_embedding_test
        PRIVATE ${mnn_SOURCE_DIR}/3rd_party/flatbuffers/include
    )
    target_link_libraries(minimind_embedding_test PRIVATE MiniMind)
    add_test(NAME minimind.embedding COMMAND minimind_embedding_test)
endif()
