cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

project(TensorX)
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

add_executable(TensorX ${SOURCE_FILES})

# Link to third_party library
# target_link_libraries(TensorX "third_party/")

# Add source files

