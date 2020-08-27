cmake_minimum_required( VERSION 3.16 FATAL_ERROR )

project( kmeans VERSION 0.0.0 )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON ) # for vim jumpping

add_executable( kmeans )
target_sources( kmeans PRIVATE  KMeans-test.cpp
                                KMeans.cpp )
