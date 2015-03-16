project(Examples)

cmake_minimum_required(VERSION 2.8)

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Function prepares name of a test executable
#    @output_name -  output variable's name
#    @filename    -  *.cpp file path
function(example_name output_name filename)
    get_filename_component(name ${filename} NAME_WE)
    set(${output_name} ${name}${EXAMPLE_EXT} PARENT_SCOPE)
endfunction()

file(GLOB EXAMPLE_CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

foreach(source ${EXAMPLE_CPP_SOURCES})
  MESSAGE( STATUS ${source} )
endforeach()

#    Build each test separately from *.cpp files
foreach(source ${EXAMPLE_CPP_SOURCES})
    example_name(EXAMPLE_NAME ${source})

    add_library(${EXAMPLE_NAME}.o EXCLUDE_FROM_ALL OBJECT ${source})
    set(EXAMPLE_OBJ_LIB $<TARGET_OBJECTS:${EXAMPLE_NAME}.o>)

    add_executable(${EXAMPLE_NAME} EXCLUDE_FROM_ALL ${EXAMPLE_OBJ_LIB})
    target_link_libraries(${EXAMPLE_NAME} purine)
    target_link_libraries(${EXAMPLE_NAME}
      ${GLOG_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      ${LIBUV_LIBRARIES}
      )

    #    output dir
    set_target_properties(${EXAMPLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

    # Targets and object libs
    set(EXAMPLE_TARGETS ${EXAMPLE_TARGETS} ${EXAMPLE_NAME})
    set(EXAMPLE_OBJ_LIBS ${EXAMPLE_OBJ_LIBS} ${EXAMPLE_OBJ_LIB})
endforeach()
