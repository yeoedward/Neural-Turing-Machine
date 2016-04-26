include (ExternalProject)

set(re2_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/re2/re2)
set(re2_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/re2/src)
set(re2_URL https://github.com/google/re2.git)
set(re2_TAG 791beff)
set(re2_BUILD ${CMAKE_BINARY_DIR}/re2/src/re2)
set(re2_LIBRARIES ${re2_BUILD}/obj/so/libre2.so)
get_filename_component(re2_STATIC_LIBRARIES ${re2_BUILD}/libre2.a ABSOLUTE)
set(re2_INCLUDES ${re2_BUILD})

# We only need re2.h in external/re2/re2/re2.h
# For the rest, we'll just add the build dir as an include dir.
set(re2_HEADERS
    "${re2_BUILD}/re2/re2.h"
)

ExternalProject_Add(re2
    PREFIX re2
    GIT_REPOSITORY ${re2_URL}
    GIT_TAG ${re2_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
)

## put re2 includes in the directory where they are expected
add_custom_target(re2_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${re2_INCLUDE_DIR}
    DEPENDS re2)

add_custom_target(re2_copy_headers_to_destination
    DEPENDS re2_create_destination_dir)

foreach(header_file ${re2_HEADERS})
    add_custom_command(TARGET re2_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${header_file} ${re2_INCLUDE_DIR})
endforeach()

ADD_LIBRARY(re2_lib STATIC IMPORTED
    DEPENDS re2)
SET_TARGET_PROPERTIES(re2_lib PROPERTIES
    IMPORTED_LOCATION ${re2_STATIC_LIBRARIES})
