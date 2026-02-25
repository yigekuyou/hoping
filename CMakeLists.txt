cmake_minimum_required(VERSION 3.19)
project(hoping LANGUAGES CXX)

find_package(xdrfile REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)

add_executable(hoping
    main.cpp
)
target_compile_features(hoping PRIVATE cxx_std_23)
include_directories(${xdrfile_INCLUDE_DIRS})
target_link_libraries(hoping
        xdrfile
        OpenCL
        m
)

include(GNUInstallDirs)

install(TARGETS hoping
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
