set (LINUX64 ON)

include_directories(${PROJECT_SOURCE_DIR}/deps/linux64/include/)
link_directories(${PROJECT_SOURCE_DIR}/deps/linux64/lib/)

set (CMAKE_CXX_COMPILE_ "g++")
set (CMAKE_CXX_FLAGS "-Dlinux -std=c++11 -ldl -lm")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set (CMAKE_CXX_FLAGS_RELEASE "-O2 -g")

set (CMAKE_C_FLAGS "-Dlinux -std=c99 -lm -lm -fPIC")
set (CMAKE_C_FLAGS_DEBUG "-O0 -g")
set (CMAKE_C_FLAGS_RELEASE "-O2 -g")