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

add_library(
    log_analyzer
    src/log_analyzer.cpp
)

target_include_directories(
    log_analyzer
    PUBLIC
    include
)

target_compile_features(
    log_analyzer
    PUBLIC
    cxx_std_17
)

target_compile_options(
    log_analyzer
    PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(
    log_analyzer
    PUBLIC
    log_parser
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

add_executable(
    log_analyzer_tests
    tests/test_log_analyzer.cpp
)

target_link_libraries(
    log_analyzer_tests
    PRIVATE
    log_analyzer
)

target_compile_options(
    log_analyzer_tests
    PRIVATE
    -Wall
    -Wextra
    -Werror
)

add_executable(
    log_analyzer_cli
    app/main.cpp
)

set_target_properties(
    log_analyzer_cli
    PROPERTIES
    OUTPUT_NAME log_analyzer
)

target_link_libraries(
    log_analyzer_cli
    PRIVATE
    log_analyzer
)

target_compile_options(
    log_analyzer_cli
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

add_test(
    NAME log_analyzer_tests
    COMMAND log_analyzer_tests
)

add_test(
    NAME log_analyzer_cli_smoke
    COMMAND
        log_analyzer_cli
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/sample.log"
        3
)

set_tests_properties(
    log_analyzer_cli_smoke
    PROPERTIES
    PASS_REGULAR_EXPRESSION "valid lines: 5"
)

set_tests_properties(
    log_parser_tests
    PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
