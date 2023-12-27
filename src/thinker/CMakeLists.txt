include_directories(${PROJECT_SOURCE_DIR}/include/thinker)
include_directories(${PROJECT_SOURCE_DIR}/src/thinker)
include_directories(${PROJECT_SOURCE_DIR}/src/thinker/c_api)
include_directories(${PROJECT_SOURCE_DIR}/src/thinker/core)

#file(GLOB C_SOURCES *.c)
file(GLOB_RECURSE C_SOURCES *.c)

add_library(thinker STATIC ${C_SOURCES})
target_compile_definitions(thinker PRIVATE "-DTHINKER_USE_VENUS=1")
