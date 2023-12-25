include_directories(${PROJECT_SOURCE_DIR}/include/wekws)
include_directories(${PROJECT_SOURCE_DIR}/src/wekws)

file(GLOB C_SOURCES *.c)
#file(GLOB_RECURSE C_SOURCES *.c)

add_library(wekws STATIC ${C_SOURCES})
add_library(wekws_so SHARED ${C_SOURCES})
target_link_libraries(wekws_so thinker hifi luna)
