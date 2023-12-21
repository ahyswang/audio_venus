include_directories(${PROJECT_SOURCE_DIR}/include/mylib)
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

add_executable(mylib_test main.c)
target_link_libraries(mylib_test mylib)
