cmake_minimum_required(VERSION 3.19)
project(hoping LANGUAGES CXX)

find_package(xdrfile REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Concurrent)
find_package(OpenMP REQUIRED)
add_executable(${PROJECT_NAME}
    main.cpp
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
include_directories(${xdrfile_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
        xdrfile
        OpenCL
        m
        Qt6::Core
        Qt6::Concurrent
        OpenMP::OpenMP_CXX
)

include(GNUInstallDirs)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<NOT:$<CONFIG:Debug>>:QT_NO_DEBUG_OUTPUT>
)
install(TARGETS hoping
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
