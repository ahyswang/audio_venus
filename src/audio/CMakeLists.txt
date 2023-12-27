include_directories(${PROJECT_SOURCE_DIR}/include/audio)
include_directories(${PROJECT_SOURCE_DIR}/src/audio)

file(GLOB C_SOURCES *.c)
#file(GLOB_RECURSE C_SOURCES *.c)

add_library(audio STATIC ${C_SOURCES})
if(${LINUX64})
    add_library(audio_so SHARED ${C_SOURCES})
    target_link_libraries(audio_so thinker hifi luna)
endif()