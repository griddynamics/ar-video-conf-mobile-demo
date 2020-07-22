#
# Copyright (C) The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.4.1)

# configure import libs
set(distribution_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../distribution)


# ${ANDROID_ABI} is handy for our purpose here. Probably this ${ANDROID_ABI} is
# the most valuable thing of this sample, the rest are pretty much normal cmake
add_library(lib_tensorflowlite SHARED IMPORTED)
set_target_properties(lib_tensorflowlite PROPERTIES IMPORTED_LOCATION
        ${distribution_DIR}/tensorflow/lib/${ANDROID_ABI}/libtensorflowlite.so)

# build application's shared lib
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_library(tensorflow_wrapper SHARED
        tensorflow_wrapper.cpp)

target_include_directories(tensorflow_wrapper PRIVATE
        ${distribution_DIR}/tensorflow/include)

target_link_libraries(tensorflow_wrapper
        android
        -ljnigraphics
        lib_tensorflowlite
        log)