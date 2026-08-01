cmake_minimum_required(VERSION 3.16)

project(LogParser LANGUAGES CXX)

add_library(
    log_parser
    src/log_parser.cpp
)

target_include_directories(
    log_parser
    PUBLIC
    include
)

target_compile_features(
    log_parser
    PUBLIC
    cxx_std_17
)

target_compile_options(
    log_parser
    PRIVATE
    -Wall
    -Wextra
    -Werror
)

add_executable(
    log_parser_tests
    tests/test_log_parser.cpp
)

target_link_libraries(
    log_parser_tests
    PRIVATE
    log_parser
)

target_compile_options(
    log_parser_tests
    PRIVATE
    -Wall
    -Wextra
    -Werror
)

enable_testing()

add_test(
    NAME log_parser_tests
    COMMAND log_parser_tests
)

set_tests_properties(
    log_parser_tests
    PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)