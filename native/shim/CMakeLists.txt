cmake_minimum_required(VERSION 3.15)
project(kaldi_native_fbank_shim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static knf, bundled into the single shim DLL; no tests/python.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kaldi-native-fbank knf-build)

add_library(kaldi_native_fbank_shim SHARED kaldi_native_fbank_shim.cc)
target_include_directories(kaldi_native_fbank_shim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../kaldi-native-fbank)
target_link_libraries(kaldi_native_fbank_shim PRIVATE kaldi-native-fbank-core)
