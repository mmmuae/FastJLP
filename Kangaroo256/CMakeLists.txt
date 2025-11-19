cmake_minimum_required(VERSION 3.20)
project(Kangaroo256 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Architectures to build" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG" CACHE STRING "" FORCE)

set(KANGAROO256_SOURCES
  Backup.cpp
  Check.cpp
  HashTable.cpp
  Kangaroo.cpp
  Merge.cpp
  Network.cpp
  PartMerge.cpp
  Thread.cpp
  Timer.cpp
  main.cpp
  SECPK1/Int.cpp
  SECPK1/IntGroup.cpp
  SECPK1/IntMod.cpp
  SECPK1/Point.cpp
  SECPK1/Random.cpp
  SECPK1/SECP256K1.cpp
)

add_executable(kangaroo256 ${KANGAROO256_SOURCES})
target_include_directories(kangaroo256 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(kangaroo256 PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(Threads REQUIRED)
target_link_libraries(kangaroo256 PRIVATE Threads::Threads)

install(TARGETS kangaroo256 RUNTIME DESTINATION bin)
