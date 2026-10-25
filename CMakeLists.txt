# ============================================================
# SpaceGo 维空弈 - CMake 构建配置
# 围棋 AI 引擎：HMGS-DP 搜索 + CA-MOE 混合专家元决策 + CUDA 加速
#
# 构建目标：
#   spacego        —— GTP 围棋引擎（默认，可对接 Sabaki/Lizzie 等）
#   spacego_train  —— 训练程序（监督/无监督/二倍体/ELO 门控）
#   SpaceGo_GUI    —— GDI+ 图形界面（仅 Windows，可选）
#
# 常用开关：
#   -DUSE_CUDA=OFF         关闭 CUDA（CPU 回退）
#   -DUSE_LIBTORCH=OFF     关闭 LibTorch（仅保留规则引擎）
#   -DBUILD_TRAIN=OFF      不构建训练程序
#   -DBUILD_GUI=OFF        不构建图形界面
#   -DLIBTORCH_ROOT=<path> 指定 LibTorch 安装路径
# ============================================================
cmake_minimum_required(VERSION 3.21)
project(SpaceGo VERSION 1.0.0 LANGUAGES C CXX)

# ------------------------------------------------------------
# 可配置选项
# ------------------------------------------------------------
option(USE_CUDA     "启用 CUDA GPU 加速"                    ON)
option(USE_LIBTORCH "启用 LibTorch 神经网络与训练"           ON)
option(BUILD_TRAIN  "构建训练程序 spacego_train"             ON)
option(BUILD_GUI    "构建 GDI+ 图形界面（仅 Windows）"       ON)
option(BUILD_TESTS  "构建单元测试（需要 GoogleTest，可自动下载）" OFF)

# LibTorch 安装路径（默认匹配 CUDA 12.4 的官方发行包）
set(LIBTORCH_ROOT "D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch"
    CACHE PATH "LibTorch 根目录")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# CUDA 支持
# ------------------------------------------------------------
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # 目标 GPU 架构：默认 GTX 1660 Ti (sm_75)，可用 -DCMAKE_CUDA_ARCHITECTURES 覆盖
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA 12.x 移除了 nvToolsExt（NVTX 已并入 Windows SDK），
    # 但 LibTorch 的 CMake 配置仍会查找 CUDA::nvToolsExt 目标，先占位避免报错
    if(NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    endif()
endif()

# ------------------------------------------------------------
# LibTorch 支持
# ------------------------------------------------------------
if(USE_LIBTORCH)
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
    find_package(Torch REQUIRED)
endif()

# ------------------------------------------------------------
# 全局编译定义与选项
# ------------------------------------------------------------
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
if(USE_CUDA)
    add_compile_definitions(USE_CUDA)
endif()
if(USE_LIBTORCH)
    add_compile_definitions(USE_LIBTORCH)
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()
if(USE_CUDA)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
endif()

# ------------------------------------------------------------
# 头文件搜索路径
# ------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)
if(USE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# ============================================================
# 共享源文件（spacego 与 spacego_train 共用，不含各自 main）
# ============================================================
set(SHARED_SOURCES
    # --- 决策层：HMGS-DP 搜索算法实现 ---
    src/decision/algorithms/calculation/diversity_calculator.cpp
    src/decision/algorithms/calculation/similarity_calculator.cpp
    src/decision/algorithms/calculation/uct_calculator.cpp
    src/decision/algorithms/config/config_manager.cpp
    src/decision/algorithms/control/search_controller.cpp
    src/decision/algorithms/data/merge_network.cpp
    src/decision/algorithms/data/node_manager.cpp
    src/decision/algorithms/execution/simulation_executor.cpp
    src/decision/algorithms/persistence/data_exporter.cpp
    src/decision/algorithms/persistence/data_importer.cpp
    src/decision/algorithms/search_engine.cpp

    # --- 决策层：引擎封装 ---
    src/decision/engine/go_engine.cpp

    # --- 决策层：CA-MOE 元决策 ---
    src/decision/search/ca_moe.cpp

    # --- 框架层 ---
    src/framework/algorithm_base.cpp
    src/framework/logger.cpp
    src/framework/performance_monitor.cpp

    # --- 感知层：棋盘表示 ---
    src/go/board/board_state.cpp

    # --- 感知层：异构计算（CPU/GPU 负载均衡） ---
    src/go/heterogeneous/device_load_balancer.cpp
    src/go/heterogeneous/heterogeneous_compute_engine.cpp
    src/go/heterogeneous/result_merger.cpp
    src/go/heterogeneous/task_decomposer.cpp
    src/go/heterogeneous/workload_analyzer.cpp

    # --- 推理层：神经网络 ---
    src/go/neural/ca_moe_expert.cpp
    src/go/neural/neural_network_cuda.cpp

    # --- 推理层：搜索加速 ---
    src/go/search/ca_moe_enhanced_integration.cpp
    src/go/search/hmgs_dp_cuda.cpp
    src/go/search/hmgs_dp_optimized.cpp
    src/go/search/spatial_topology_accelerator.cpp
    src/go/search/spatial_topology_calculator.cpp

    # --- 训练体系 ---
    src/go/training/auto_tuner.cpp
    src/go/training/cli_tuner.cpp
    src/go/training/diploid_trainer.cpp
    src/go/training/elo_gated_trainer.cpp
    src/go/training/experience_buffer.cpp
    src/go/training/experience_collector.cpp
    src/go/training/multi_dim_evaluator.cpp
    src/go/training/param_registry.cpp
    src/go/training/supervised_pretrain.cpp
    src/go/training/unsupervised_self_reinforcement.cpp

    # --- 搜索算法核心（HMGS-DP 节点与主循环） ---
    src/network/algorithms/hmgs_dp.cpp
    src/network/algorithms/hmgs_node.cpp

    # --- 通信协议（GTP） ---
    src/network/protocol/gtp_protocol.cpp

    # --- 感知层：CUDA 棋盘与哈希 ---
    src/perception/board/go_board_cuda.cpp
    src/perception/board/zobrist_hash.cpp
)

# ============================================================
# CUDA 源文件
# ============================================================
set(CUDA_SOURCES "")
if(USE_CUDA)
    set(CUDA_SOURCES
        src/perception/cuda/go_board_kernels.cu
    )
endif()

# ============================================================
# 目标1：spacego —— GTP 围棋引擎
# ============================================================
add_executable(spacego
    src/main.cpp
    ${SHARED_SOURCES}
    ${CUDA_SOURCES}
)

if(USE_LIBTORCH)
    target_link_libraries(spacego "${TORCH_LIBRARIES}")
endif()
if(USE_CUDA)
    # NVML 用于 GPU 利用率/显存监控（device_load_balancer、performance_monitor）
    target_link_libraries(spacego CUDA::nvml)
endif()

# ============================================================
# 目标2：spacego_train —— 训练程序
# ============================================================
if(BUILD_TRAIN)
    add_executable(spacego_train
        src/train_main.cpp
        ${SHARED_SOURCES}
        ${CUDA_SOURCES}
    )

    if(USE_LIBTORCH)
        target_link_libraries(spacego_train "${TORCH_LIBRARIES}")
    endif()
    if(USE_CUDA)
        target_link_libraries(spacego_train CUDA::nvml)
    endif()
endif()

# ============================================================
# 目标3：SpaceGo_GUI —— 图形界面（可选，仅 Windows）
# ============================================================
if(BUILD_GUI AND WIN32)
    add_subdirectory(GUI)
endif()

# ============================================================
# 目标4：单元测试（可选）
# ============================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
