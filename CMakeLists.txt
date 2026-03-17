cmake_minimum_required(VERSION 4.2.3)

project(Tenzo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(Tenzo
    src/tensor.cpp
)

target_include_directories(Tenzo
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(main main.cpp)

target_link_libraries(main
    PRIVATE Tenzo
)
