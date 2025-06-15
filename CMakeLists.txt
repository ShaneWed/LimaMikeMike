cmake_minimum_required(VERSION 3.30)
project(LimaMikeMike)

set(CMAKE_CXX_STANDARD 20)

add_executable(LimaMikeMike src/main.cpp
        src/MultiLayerPerceptron.cpp
        src/MultiLayerPerceptron.h
        src/Layer.cpp
        src/Layer.h)
