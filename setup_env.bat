@echo off
setlocal enabledelayedexpansion

REM ---- Configuration ----
set "PYTHON_VERSION=3.11"
set "VENV_DIR=.venv"

REM ---- Resolve CUDA_HOME / nvcc ----
REM Must use CUDA 12.8+ for sm_120 support
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

echo === CUDA vars ===
echo CUDA_HOME=%CUDA_HOME%
echo CUDA_PATH=%CUDA_PATH%

if exist "%CUDA_HOME%\bin\nvcc.exe" goto cuda_ok

REM If the expected path doesn't have nvcc, try to find nvcc via PATH.
where nvcc >nul 2>nul
if errorlevel 1 goto cuda_missing

for /f "delims=" %%I in ('where nvcc') do (
    set "NVCC_PATH=%%I"
    goto found_nvcc
)

:found_nvcc
REM NVCC_PATH points to ...\bin\nvcc.exe
for %%D in ("%NVCC_PATH%") do set "NVCC_DIR=%%~dpD"
REM NVCC_DIR ends with \bin\, so CUDA_HOME is its parent
for %%D in ("%NVCC_DIR%\..") do set "CUDA_HOME=%%~fD"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

echo Found nvcc at: %NVCC_PATH%
echo Using CUDA_HOME: %CUDA_HOME%
goto cuda_ok

:cuda_missing
echo ERROR: nvcc not found.
echo - Install CUDA Toolkit 12.8+ (recommended for sm_120) OR
echo - Set CUDA_HOME to your CUDA install folder OR
echo - Ensure nvcc.exe is on PATH
exit /b 1

:cuda_ok

echo === Creating virtual environment (Python %PYTHON_VERSION%) ===
py -%PYTHON_VERSION% -m venv %VENV_DIR%
if errorlevel 1 (
    echo Failed to create venv
    exit /b 1
)

echo === Activating virtual environment ===
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate venv
    exit /b 1
)

echo === Upgrading pip ===
python -m pip install -U pip
if errorlevel 1 (
    echo Failed to upgrade pip
    exit /b 1
)

echo === Installing PyTorch (stable cu128) ===
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo Failed to install PyTorch stable cu128
    exit /b 1
)

echo === Verifying torch CUDA ===
python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('arch list:', torch.cuda.get_arch_list()); x=torch.randn(1, device='cuda'); print('cuda tensor ok:', x)"
if errorlevel 1 (
    echo Torch CUDA verification failed
    exit /b 1
)

echo === Installing requirements ===
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements
    exit /b 1
)

echo === Environment setup complete ===
endlocal
pause