# =============================================================================
# SpaceGo 围棋AI - 单元测试 CMakeLists.txt
# 使用 Google Test 框架进行测试
# =============================================================================

# ---- 获取 Google Test ----
# 优先尝试 find_package（系统已安装GTest），失败则使用 FetchContent 自动下载
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest 未在系统中找到，使用 FetchContent 从 GitHub 下载...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    # 设置 gtest_force_shared_crt 以避免 Windows MSVC 运行时库冲突
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    message(STATUS "GTest v1.14.0 下载并构建完成")
else()
    message(STATUS "使用系统已安装的 GTest")
endif()

# ---- 收集测试源文件 ----
# 测试用例源文件
set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_go_board_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_search.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_gtp_protocol.cpp
)

# ---- 收集被测试的核心源文件（复用主项目的源文件集合） ----
# 注意：排除 src/main.cpp，因为测试有自己的入口

file(GLOB_RECURSE COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/framework/*.cpp
)

file(GLOB_RECURSE GO_ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/decision/**/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/go/**/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/perception/board/*.cpp
)

file(GLOB_RECURSE NETWORK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/network/**/*.cpp
)

# ---- CUDA 源文件 ----
if(USE_CUDA)
    file(GLOB CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/perception/cuda/*.cu
    )
else()
    set(CUDA_SOURCES "")
endif()

# ---- 创建测试可执行文件 ----
add_executable(spacego_test
    ${TEST_SOURCES}
    ${COMMON_SOURCES}
    ${GO_ENGINE_SOURCES}
    ${NETWORK_SOURCES}
    ${CUDA_SOURCES}
)

# ---- 头文件包含路径 ----
target_include_directories(spacego_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src  # 部分内部头文件需要
)

# ---- 编译定义（与主项目保持一致） ----
target_compile_definitions(spacego_test PRIVATE
    $<$<BOOL:${USE_CUDA}>:USE_CUDA>
    $<$<BOOL:${USE_LIBTORCH}>:USE_LIBTORCH>
    $<$<BOOL:${USE_GUI}>:USE_GUI>
)

# ---- Windows 编译选项 ----
if(MSVC)
    target_compile_options(spacego_test PRIVATE /utf-8 /W3 /EHsc /bigobj /FS)
endif()

# ---- CUDA 支持 ----
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_include_directories(spacego_test PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(spacego_test PRIVATE
        CUDA::cudart
        CUDA::nvml
    )

    set_target_properties(spacego_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(WIN32)
        target_compile_options(spacego_test PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
endif()

# ---- LibTorch 支持 ----
if(USE_LIBTORCH)
    if(USE_CUDA AND NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()

    find_package(Torch REQUIRED PATHS "${LIBTORCH_ROOT}" NO_DEFAULT_PATH)

    target_include_directories(spacego_test PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(spacego_test PRIVATE
        ${TORCH_LIBRARIES}
    )

    if(WIN32)
        target_compile_definitions(spacego_test PRIVATE
            TORCH_API_INCLUDE_WITHOUTtorch
        )
    endif()
endif()

# ---- 链接库 ----
target_link_libraries(spacego_test PRIVATE
    GTest::gtest_main      # 包含 Google Test 的 main() 入口
    Threads::Threads
    $<$<PLATFORM_ID:Windows>:ws2_32;winmm>
)

# ---- CTest 集成 ----
include(GoogleTest)
gtest_discover_tests(spacego_test
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        LABELS "unit"
        TIMEOUT 120
)