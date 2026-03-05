@echo off
setlocal enabledelayedexpansion

set "VENV_DIR=.venv"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

echo === Activating virtual environment ===
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate venv
    exit /b 1
)

if "%1"=="clean" (
    echo === Running clean ===
    python setup.py clean
    exit /b 0
)

if "%1"=="rebuild" (
    echo === Running rebuild ===
    python setup.py clean
)

set "TORCH_CUDA_ARCH_LIST=12.0+PTX"

echo === Building extension ===
python setup.py build_ext --inplace
if errorlevel 1 (
    echo Failed to build extension
    exit /b 1
)

echo === Build complete ===
endlocal
pause