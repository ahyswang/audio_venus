include_directories(${PROJECT_SOURCE_DIR}/include/snoring_detect)
include_directories(${PROJECT_SOURCE_DIR}/src/snoring_detect)
include_directories(${PROJECT_SOURCE_DIR}/include/audio)
include_directories(${PROJECT_SOURCE_DIR}/src/audio)

file(GLOB C_SOURCES *.c)
#file(GLOB_RECURSE C_SOURCES *.c)

add_library(snoring_detect STATIC ${C_SOURCES})
add_library(snoring_detect_so SHARED ${C_SOURCES})
target_link_libraries(snoring_detect_so audio thinker hifi luna)
