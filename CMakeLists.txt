cmake_minimum_required(VERSION 3.20)
project(Kangaroo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(DEFAULT_GPU_BACKEND_METAL OFF)
if(APPLE)
  set(DEFAULT_GPU_BACKEND_METAL ON)
endif()
option(GPU_BACKEND_METAL "Build Metal backend" ${DEFAULT_GPU_BACKEND_METAL})

set(DEFAULT_GPU_BACKEND_CUDA OFF)
if(NOT APPLE)
  set(DEFAULT_GPU_BACKEND_CUDA ON)
endif()
option(GPU_BACKEND_CUDA "Build CUDA backend" ${DEFAULT_GPU_BACKEND_CUDA})

option(BUILD_METAL_TESTS "Build Metal unit tests" ON)

include(GNUInstallDirs)

list(APPEND KANGAROO_SOURCES
  main.cpp
  Timer.cpp
  Backup.cpp
  Check.cpp
  HashTable.cpp
  Kangaroo.cpp
  Merge.cpp
  Network.cpp
  PartMerge.cpp
  Thread.cpp
  SECPK1/Int.cpp
  SECPK1/IntGroup.cpp
  SECPK1/IntMod.cpp
  SECPK1/Point.cpp
  SECPK1/Random.cpp
  SECPK1/SECP256K1.cpp
  GPU/BackendFactory.cpp
)

add_executable(kangaroo ${KANGAROO_SOURCES})
target_include_directories(kangaroo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(kangaroo PRIVATE "$<$<CONFIG:Debug>:DEBUG>")

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
if(MSVC)
  set(KANGAROO_WARNING_FLAGS /W4)
  if(WARNINGS_AS_ERRORS)
    list(APPEND KANGAROO_WARNING_FLAGS /WX)
  endif()
  target_compile_options(kangaroo PRIVATE ${KANGAROO_WARNING_FLAGS})
else()
  set(KANGAROO_WARNING_FLAGS -Wall -Wextra -Wpedantic)
  if(WARNINGS_AS_ERRORS)
    list(APPEND KANGAROO_WARNING_FLAGS -Werror)
  endif()
  target_compile_options(kangaroo PRIVATE ${KANGAROO_WARNING_FLAGS})
endif()

if(GPU_BACKEND_METAL)
  enable_language(OBJCXX OPTIONAL)
  if(NOT CMAKE_OBJCXX_COMPILER)
    message(FATAL_ERROR "Metal backend requested but no Objective-C++ compiler is available")
  endif()

  target_compile_definitions(kangaroo PRIVATE GPU_BACKEND_METAL)
  target_sources(kangaroo PRIVATE GPU/metal/MetalBackend.mm)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/GPU/metal/kernels.metal
    ${CMAKE_CURRENT_BINARY_DIR}/GPU/metal/kernels.metal
    COPYONLY)

  if(APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(kangaroo PRIVATE
      "${METAL_FRAMEWORK}"
      "${METALKIT_FRAMEWORK}"
      "${FOUNDATION_FRAMEWORK}")
    set_source_files_properties(GPU/metal/MetalBackend.mm PROPERTIES
      LANGUAGE OBJCXX)
  else()
    message(WARNING "Metal backend enabled on a non-Apple platform; build may fail due to missing frameworks")
  endif()
endif()

if(GPU_BACKEND_CUDA)
  target_compile_definitions(kangaroo PRIVATE GPU_BACKEND_CUDA)
  # CUDA integration is expected to be supplied by existing project files.
endif()

if(BUILD_METAL_TESTS AND GPU_BACKEND_METAL)
  add_executable(MetalPackingTests
    tests/MetalPackingTests.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp)
  target_include_directories(MetalPackingTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(MetalPackingTests PRIVATE GPU_BACKEND_METAL)
  target_compile_options(MetalPackingTests PRIVATE ${KANGAROO_WARNING_FLAGS})

  add_executable(MetalDistinguishedPointTests
    tests/MetalDistinguishedPointTests.cpp
    SECPK1/Int.cpp
    SECPK1/IntGroup.cpp
    SECPK1/IntMod.cpp
    SECPK1/Random.cpp
    Timer.cpp)
  target_include_directories(MetalDistinguishedPointTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(MetalDistinguishedPointTests PRIVATE GPU_BACKEND_METAL)
  target_compile_options(MetalDistinguishedPointTests PRIVATE ${KANGAROO_WARNING_FLAGS})

  add_executable(MetalU256Smoke
    tests/MetalU256Smoke.mm)
  target_compile_definitions(MetalU256Smoke PRIVATE GPU_BACKEND_METAL)
  target_compile_options(MetalU256Smoke PRIVATE ${KANGAROO_WARNING_FLAGS})

  if(APPLE)
    target_link_libraries(MetalPackingTests PRIVATE
      "${FOUNDATION_FRAMEWORK}"
      "${METAL_FRAMEWORK}"
      "${METALKIT_FRAMEWORK}")
    target_link_libraries(MetalDistinguishedPointTests PRIVATE
      "${FOUNDATION_FRAMEWORK}"
      "${METAL_FRAMEWORK}"
      "${METALKIT_FRAMEWORK}")
    target_link_libraries(MetalU256Smoke PRIVATE
      "${FOUNDATION_FRAMEWORK}"
      "${METAL_FRAMEWORK}"
      "${METALKIT_FRAMEWORK}")
  endif()
endif()

install(TARGETS kangaroo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
