cmake_minimum_required(VERSION 3.19)
project(code LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core LinguistTools)
find_package(xdrfile REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)

qt_standard_project_setup()

qt_add_executable(code
    main.cpp
    kernel.cl
)

qt_add_translations(
    TARGETS code
    TS_FILES code_zh_CN.ts
)
include_directories(${xdrfile_INCLUDE_DIRS})
target_link_libraries(code
    PRIVATE
        Qt::Core
        xdrfile
        OpenCL
        m
)

include(GNUInstallDirs)

install(TARGETS code
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
