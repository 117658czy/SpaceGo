# 构建指南

## 一、环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 10/11 / Linux | GUI 仅支持 Windows |
| 编译器 | MSVC 2022 (v143) / GCC 11+ | 需支持 C++17 |
| CMake | 3.21+ | 构建系统 |
| CUDA Toolkit | 12.x（推荐 12.4） | 可选，GPU 加速 |
| LibTorch | 2.6.0+cu124 | 可选，神经网络推理与训练 |
| NVIDIA 驱动 | 支持 CUDA 12.4 | 包含 NVML，用于 GPU 监控 |

### 依赖获取

- **CUDA Toolkit**：https://developer.nvidia.com/cuda-downloads
- **LibTorch**（与 CUDA 12.4 匹配）：
  ```
  https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.6.0%2Bcu124.zip
  ```
  解压到任意目录（例如 `D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch`）。

## 二、构建

### 1. 标准构建（Windows / Visual Studio）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_BUILD_TYPE=Release `
      -DLIBTORCH_ROOT="D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch"
cmake --build build --config Release
```

产物位于 `build/Release/`：`spacego.exe`、`spacego_train.exe`、`SpaceGo_GUI.exe`。

### 2. Ninja 构建（更快）

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DLIBTORCH_ROOT="D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch"
cmake --build build
```

使用 Ninja 前请先进入 **Developer Command Prompt for VS**（保证 `cl.exe` 可用）。

### 3. 无 GPU / 无 LibTorch 的最小构建

仅验证引擎主体（规则搜索 + GTP）：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DUSE_CUDA=OFF -DUSE_LIBTORCH=OFF -DBUILD_TRAIN=OFF -DBUILD_GUI=OFF
cmake --build build --config Release
```

### 4. 单元测试（可选）

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

> 未检测到系统 GoogleTest 时，会通过 FetchContent 自动下载 v1.14.0（需联网）。

## 三、常用 CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `USE_CUDA` | ON | 启用 CUDA 编译与 GPU 加速 |
| `USE_LIBTORCH` | ON | 启用 LibTorch（神经网络与训练） |
| `BUILD_TRAIN` | ON | 构建 `spacego_train` |
| `BUILD_GUI` | ON | 构建 GDI+ 图形界面（仅 Windows） |
| `BUILD_TESTS` | OFF | 构建单元测试 |
| `LIBTORCH_ROOT` | D:/libtorch-.../libtorch | LibTorch 安装路径 |
| `CMAKE_CUDA_ARCHITECTURES` | 75 | 目标 GPU 架构（如 `86`、`89`） |

## 四、运行

```powershell
# GTP 引擎（模型可省略，省略时使用规则评估）
.\spacego.exe --gtp --simulations=1600
.\spacego.exe --gtp --model=models/ca_moe.pt

# 交互式命令行
.\spacego.exe --interactive

# 训练
.\spacego_train.exe --mode=supervised
.\spacego_train.exe --mode=elo-gated --elo-rounds=300 --output-dir=./training_output
```

### GTP 前端对接

以 Sabaki 为例：引擎路径填写 `spacego.exe`，启动参数填写 `--gtp`（如需模型：`--gtp --model=<路径>`）。

## 五、常见问题

**Q1：链接时报找不到 `nvmlInit_v2` / NVML 相关符号**
`USE_CUDA=ON` 时会启用 NVML GPU 监控，需链接 NVML（CMake 已通过 `CUDA::nvml` 自动处理）。若 CUDA 安装不含 NVML，请安装完整版 CUDA Toolkit 或关闭 GPU 监控。

**Q2：路径含中文/空格导致 CUDA 编译器识别失败**
典型报错：

```
c1xx : fatal error C1083: 无法打开源文件: "E:\SpaceGo ": No such file or directory
  ... CMakeDetermineCUDACompiler.cmake ... enable_language
```

原因是 CMake 探测 CUDA 编译器时，`nvcc` 经 MSBuild 传参时未对含空格/中文的路径加引号，路径被按空格拆开。
**解决**：使用纯 ASCII 路径。若必须保留中文源码路径，可建立目录联结（junction，无需管理员权限）后从 ASCII 路径构建：

```powershell
# 建立联结：E:\sg_oss -> 实际源码目录
New-Item -ItemType Junction -Path "E:\sg_oss" -Target "E:\SpaceGo 维空弈\opensource"

# 从 ASCII 路径配置与构建（构建产物仍落在源码目录的 build/ 下）
cmake -S E:\sg_oss -B E:\sg_oss\build -G "Visual Studio 17 2022" -A x64 `
      -DLIBTORCH_ROOT="D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch"
cmake --build E:\sg_oss\build --config Release

# 验证完成后删除联结（不会删除源码）
cmd /c rmdir "E:\sg_oss"
```

**Q3：编译时报 PDB 写入失败**
同样源于中文路径。将源码或构建目录放在纯英文路径下（如 `D:/build_spacego`）。

**Q4：找不到 `cl.exe`**
使用 “Developer Command Prompt for VS 2022” 运行 CMake，或先执行 `vcvarsall.bat x64`。

**Q5：找不到 `torch/torch.h`**
检查 `LIBTORCH_ROOT` 是否指向解压后的 `libtorch` 目录（其下应含 `include/`、`lib/`、`share/cmake/Torch/`）。

**Q6：CUDA 架构不匹配（如 40 系显卡）**
配置时指定 `-DCMAKE_CUDA_ARCHITECTURES=89`。

## 六、运行期依赖

Windows 下运行时需将 LibTorch 的 DLL 目录（`<LIBTORCH_ROOT>/lib`）以及 CUDA 运行库加入 `PATH`。
