##
#  CMake script for the step-1 tutorial program with VTK support
##

cmake_minimum_required(VERSION 3.13.4)

# Set the name of the project and target:
set(TARGET "main")

# Declare source files
set(TARGET_SRC
  ${TARGET}.cpp
  src/graph.cpp
  src/MG_Level.cpp
  src/multigridpreconditioner.cpp
  src/poisson.cpp
)

# --- Deal.II setup ---
find_package(deal.II 9.6.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
endif()

add_executable(${TARGET} ${TARGET_SRC})

target_include_directories(${TARGET}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER mpicxx)

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_setup_target(${TARGET})
