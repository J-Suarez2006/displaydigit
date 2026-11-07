cmake_minimum_required(VERSION 3.26)
project(main C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

find_package(SDL3 REQUIRED CONFIG)

add_executable(main)

target_sources(main
PRIVATE
    main.cpp
    src/drawpixels.cpp
    src/parsemnist.cpp
    src/NN.cpp
)

target_link_libraries(main SDL3::SDL3)
