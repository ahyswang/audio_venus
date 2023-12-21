include_directories(${PROJECT_SOURCE_DIR}/include/mylib)

add_library(mylib STATIC mylib.c)
add_library(mylib_so SHARED mylib.c)
