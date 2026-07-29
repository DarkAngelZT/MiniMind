# MiniMind keeps MNN's differentiable graph, high-level NN modules and
# optimizers. Dataset readers, sample models and training executables are
# intentionally excluded from this vendored build.
option(MNN_TRAIN_DEBUG "Enable MNN Train Grad Debug" OFF)

include_directories(${CMAKE_CURRENT_LIST_DIR}/source/grad)
include_directories(${CMAKE_CURRENT_LIST_DIR}/source/optimizer)
include_directories(${CMAKE_CURRENT_LIST_DIR}/source/transformer)
include_directories(${CMAKE_CURRENT_LIST_DIR}/source/nn)

file(GLOB GRAD        ${CMAKE_CURRENT_LIST_DIR}/source/grad/*)
file(GLOB OPTIMIZER   ${CMAKE_CURRENT_LIST_DIR}/source/optimizer/*)
file(GLOB TRANSFORMER ${CMAKE_CURRENT_LIST_DIR}/source/transformer/*)
file(GLOB NNFILES     ${CMAKE_CURRENT_LIST_DIR}/source/nn/*)

if(MNN_TRAIN_DEBUG)
    add_definitions(-DMNN_TRAIN_DEBUG)
endif()

add_library(MNNTrain OBJECT
    ${GRAD}
    ${OPTIMIZER}
    ${TRANSFORMER}
    ${NNFILES}
)

# MNN's root build expects this target when MNN_BUILD_TRAIN is enabled.
# The utility library normally contains image datasets and demo models; a
# dummy object preserves the upstream target contract without shipping them.
add_library(MNNTrainUtils OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/../../cmake/dummy.cpp
)
