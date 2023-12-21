include_directories(${PROJECT_SOURCE_DIR}/include/wekws)
include_directories(${PROJECT_SOURCE_DIR}/src/wekws)
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

file(GLOB C_SOURCES *.c)

add_executable(wekws_test ${C_SOURCES})
target_link_libraries(wekws_test wekws)
