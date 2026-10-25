# ============================================================================
# SpaceGo GUI - PowerShell构建脚本
# 按照构建说明.md执行
# ============================================================================

Write-Host "========================================" -ForegroundColor Green
Write-Host "SpaceGo GUI - 构建项目" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查是否在VS开发者命令提示符中
if (-not (Test-Path "env:VSINSTALLDIR")) {
    Write-Host "错误: 请在 'x64 Native Tools Command Prompt for VS 2022' 中运行此脚本！" -ForegroundColor Red
    Write-Host "请按以下步骤操作：" -ForegroundColor Yellow
    Write-Host "1. 打开开始菜单" -ForegroundColor Yellow
    Write-Host "2. 搜索 'x64 Native Tools Command Prompt for VS 2022'" -ForegroundColor Yellow
    Write-Host "3. 在那个命令提示符中运行：powershell -ExecutionPolicy Bypass -File build.ps1" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "[1/4] 切换到项目目录..." -ForegroundColor Green
Set-Location $PSScriptRoot
Write-Host "当前目录: $(Get-Location)" -ForegroundColor Cyan

Write-Host ""
Write-Host "[2/4] 创建并进入build目录..." -ForegroundColor Green
if (Test-Path build) {
    Remove-Item build -Recurse -Force
    Write-Host "已删除旧的build目录" -ForegroundColor Yellow
}
New-Item -ItemType Directory build | Out-Null
Set-Location build
Write-Host "已进入build目录" -ForegroundColor Cyan

Write-Host ""
Write-Host "[3/4] 配置CMake..." -ForegroundColor Green
$cmakePath = "D:\源程序\cmake-4.2.3-windows-x86_64\cmake-4.2.3-windows-x86_64\bin\cmake.exe"
&amp; $cmakePath .. -G "Visual Studio 17 2022" -A x64

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "CMake配置失败！" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "CMake配置成功！" -ForegroundColor Green

Write-Host ""
Write-Host "[4/4] 编译项目..." -ForegroundColor Green
&amp; $cmakePath --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "编译失败！" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "编译成功！" -ForegroundColor Green
Write-Host "可执行文件位置: ..\bin\Release\SpaceGo_GUI.exe" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "按任意键运行程序..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "运行程序..." -ForegroundColor Green
Set-Location ..
Start-Process ".\bin\Release\SpaceGo_GUI.exe"
