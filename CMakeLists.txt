cmake_minimum_required(VERSION 3.5)
project(demo)
set(CMAKE_CXX_STANDARD 11)
set(TENSORFLOW_DIR /home/mazx/git/tensorflow)
<<<<<<< HEAD
set(CMAKE_CXX_FLAGS "")
=======
set(CMAKE_CXX_FLAGS "-g -O3 -Wl,--version-script,${TENSORFLOW_DIR}/tensorflow/c/version_script.lds")
>>>>>>> 1b8b58b298ee364146a93f1c6f9fb9edc3851aba
#set(CMAKE_CXX_FLAGS "-g -O3 -Wl,-exported_symbols_list,${TENSORFLOW_DIR}/tensorflow/tf_exported_symbols.lds,--version-script,${TENSORFLOW_DIR}/tensorflow/tf_version_script.lds")
include_directories(${TENSORFLOW_DIR})
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/include)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/eigen)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/public)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/absl)

link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib)
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11)
link_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow)

add_executable(tftest src/main.cc)
<<<<<<< HEAD
add_executable(tfrun src/test.cc)
=======
>>>>>>> 1b8b58b298ee364146a93f1c6f9fb9edc3851aba
target_link_libraries(tftest tensorflow_cc tensorflow_framework)
target_link_libraries(tfrun tensorflow_cc tensorflow_framework)
