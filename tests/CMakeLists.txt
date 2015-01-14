project(Test)

cmake_minimum_required(VERSION 2.8)

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})


# Function prepares name of a test executable
#    @output_name -  output variable's name
#    @filename    -  test_*.cpp file path
function(test_name output_name filename)
    get_filename_component(name ${filename} NAME_WE)
    set(${output_name} ${name}${TEST_EXT} PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE TEST_CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "test_*.cpp")
list(REMOVE_ITEM TEST_CPP_SOURCES test_main.cpp)

foreach(source ${TEST_CPP_SOURCES})
  MESSAGE( STATUS ${source} )
endforeach()

add_library(test_main.o EXCLUDE_FROM_ALL OBJECT test_main.cpp)

#    Build each test separately from *.cpp files
foreach(source ${TEST_CPP_SOURCES})
    test_name(TEST_NAME ${source})

    add_library(${TEST_NAME}.o EXCLUDE_FROM_ALL OBJECT ${source})
    set(TEST_OBJ_LIB $<TARGET_OBJECTS:${TEST_NAME}.o>)

    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_OBJ_LIB} $<TARGET_OBJECTS:test_main.o>)
    target_link_libraries(${TEST_NAME} purine)
    target_link_libraries(${TEST_NAME}
      ${GLOG_LIBRARIES}
      ${LIBUV_LIBRARIES}
      )

    #    output dir
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

    # Targets and object libs
    set(TEST_TARGETS ${TEST_TARGETS} ${TEST_NAME})
    set(TEST_OBJ_LIBS ${TEST_OBJ_LIBS} ${TEST_OBJ_LIB})
endforeach()

# Build a compound test excluded from the ALL target
add_executable(test_all EXCLUDE_FROM_ALL ${TEST_OBJ_LIBS} $<TARGET_OBJECTS:test_main.o>)
target_link_libraries(test_all purine)
target_link_libraries(test_all
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${GLOG_LIBRARIES}
  ${LIBUV_LIBRARIES}
  )
add_dependencies(test_all ${TEST_TARGETS})

# Test command
add_custom_target(run_test COMMAND test_all)
