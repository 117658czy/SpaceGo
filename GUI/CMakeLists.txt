# ============================================================
# SpaceGo 维空弈 - GUI 构建配置（GDI+ 图形界面）
#
# 可独立构建（直接 cmake 本目录），也可作为根工程子目录被引
# 入（-DBUILD_GUI=ON）。界面复用引擎的日志实现，因此需要同时
# 编译 ../src/framework/logger.cpp。
# ============================================================
cmake_minimum_required(VERSION 3.15)
project(SpaceGo_GUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GUI_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")
set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# 根工程为引擎全局定义了 WIN32_LEAN_AND_MEAN，该宏会让 windows.h 不再引入
# objidl.h，导致 GDI+ 的 gdiplus.h 缺失 IStream / PROPID 等声明而编译失败。
# 界面代码不依赖该宏，故在本目录范围移除（引擎自身仍保持定义）。
remove_definitions(-DWIN32_LEAN_AND_MEAN)

# 界面源码 + 引擎头文件目录
include_directories(
    ${GUI_DIR}/include
    ${ROOT_DIR}/include
)

# 源文件列表
set(SOURCES
    main.cpp
    src/utils/color.cpp
    src/utils/font.cpp
    src/utils/logger.cpp
    src/core/window.cpp
    src/core/renderer.cpp
    src/core/animation.cpp
    src/models/piece.cpp
    src/models/board.cpp
    src/models/game.cpp
    src/components/splash_window.cpp
    src/components/button.cpp
    src/components/tab.cpp
    src/components/board.cpp
    src/components/toolbar.cpp
    src/components/game_info_panel.cpp
    src/components/main_window.cpp
    # 复用引擎日志实现（GUI 的 logger.h 基于 framework/logger_compat.h）
    ${ROOT_DIR}/src/framework/logger.cpp
)

add_executable(SpaceGo_GUI WIN32 ${SOURCES})

target_link_libraries(SpaceGo_GUI gdiplus user32 gdi32)

if(MSVC)
    target_compile_options(SpaceGo_GUI PRIVATE /utf-8)
endif()

set_target_properties(SpaceGo_GUI PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${GUI_DIR}/bin
)
