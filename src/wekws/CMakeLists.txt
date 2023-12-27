include_directories(${PROJECT_SOURCE_DIR}/include/wekws)
include_directories(${PROJECT_SOURCE_DIR}/src/wekws)
include_directories(${PROJECT_SOURCE_DIR}/include/audio)
include_directories(${PROJECT_SOURCE_DIR}/src/audio)

file(GLOB C_SOURCES *.c)
#file(GLOB_RECURSE C_SOURCES *.c)

add_library(wekws STATIC ${C_SOURCES})
if(${LINUX64})
    add_library(wekws_so SHARED ${C_SOURCES})
    target_link_libraries(wekws_so thinker hifi luna)
endif()