@echo off
setlocal enabledelayedexpansion

REM Hybrid Assistant Launcher
REM This script creates a virtual environment, installs dependencies, and runs the app

echo.
echo ============================================
echo   Hybrid Assistant - Launcher
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and add it to your PATH
    pause
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Define venv path
set VENV_DIR=%SCRIPT_DIR%venv
set PYTHON=%VENV_DIR%\Scripts\python.exe
set PIP=%VENV_DIR%\Scripts\pip.exe

REM Parse optional arg: --reinstall to force dependencies reinstall
set FORCE_INSTALL=0
if "%~1"=="--reinstall" set FORCE_INSTALL=1
if "%~1"=="--force" set FORCE_INSTALL=1

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    <nul set /p ="Creating virtual environment... "
    python -m venv "%VENV_DIR%" >nul 2>&1
    if errorlevel 1 (
        echo Failed
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    ) else (
        echo Done
    )
)

REM Activate venv and install/upgrade requirements (skip if already done)
echo.
echo Installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"

if exist "%VENV_DIR%\.installed" (
    if "%FORCE_INSTALL%"=="1" (
        echo Reinstall forced by --reinstall flag; running install steps...
    ) else (
        echo Dependencies already installed; skipping installation. (Use --reinstall to force)
        goto after_install
    )
)

REM Use PowerShell to run pip commands with a simple spinner and capture output to logs
<nul set /p ="Upgrading pip... "
powershell -NoProfile -Command "& { $exe = '%PYTHON%'; $args = '-m pip install --upgrade pip'; $log = 'pip_upgrade.log'; $psi = New-Object System.Diagnostics.ProcessStartInfo($exe, $args); $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false; $p = New-Object System.Diagnostics.Process; $p.StartInfo = $psi; $p.Start() | Out-Null; while (-not $p.HasExited) { Write-Host -NoNewline '.'; Start-Sleep -Milliseconds 300 }; $out = $p.StandardOutput.ReadToEnd() + $p.StandardError.ReadToEnd(); $out | Out-File -FilePath $log -Encoding utf8; if ($p.ExitCode -ne 0) { Write-Host ' Failed'; exit $p.ExitCode } else { Write-Host ' Done' } }"

<nul set /p ="Installing requirements (this can take several minutes)... "
powershell -NoProfile -Command "& { $exe = '%PYTHON%'; $args = '-m pip install -r requirements.txt'; $log = 'install_requirements.log'; $psi = New-Object System.Diagnostics.ProcessStartInfo($exe, $args); $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false; $p = New-Object System.Diagnostics.Process; $p.StartInfo = $psi; $p.Start() | Out-Null; while (-not $p.HasExited) { Write-Host -NoNewline '.'; Start-Sleep -Milliseconds 300 }; $out = $p.StandardOutput.ReadToEnd() + $p.StandardError.ReadToEnd(); $out | Out-File -FilePath $log -Encoding utf8; if ($p.ExitCode -ne 0) { Write-Host ' Failed'; Write-Host 'WARNING: Some dependencies may have failed to install. See ' + $log; exit 0 } else { Write-Host ' Done' } }"

:after_install
REM Mark installation done if not forced to reinstall
if exist "%VENV_DIR%" if not "%FORCE_INSTALL%"=="1" type nul > "%VENV_DIR%\.installed"

REM Run pre-flight checks
echo.
echo Running pre-flight checks...
%PYTHON% startup_check.py
if errorlevel 1 (
    echo.
    echo Pre-flight checks failed. Checking network connectivity...
    REM Quick connectivity check: ping a reliable DNS server
    ping -n 1 8.8.8.8 >nul 2>&1
    if errorlevel 1 (
        echo No internet detected -- continuing in offline mode.
        set TRANSFORMERS_OFFLINE=1
        REM Continue startup despite pre-flight failure when offline
    ) else (
        echo Internet detected. Please fix pre-flight issues and try again.
        pause
        exit /b 1
    )
)

REM Show mode selection menu
echo.
echo ============================================
echo   Select Running Mode
echo ============================================
echo.
echo 1. CPU Mode (safest, slower)
echo 2. CUDA Mode (GPU acceleration, requires NVIDIA)
echo 3. Hybrid Mode (auto-detect, recommended)
echo 4. Exit
echo.

set /p MODE="Enter your choice (1-4): "

if "%MODE%"=="1" goto cpu_mode
if "%MODE%"=="2" goto cuda_mode
if "%MODE%"=="3" goto hybrid_mode
if "%MODE%"=="4" goto exit_script

echo Invalid choice. Defaulting to CPU mode...
goto cpu_mode

:cpu_mode
echo.
echo Starting Hybrid Assistant in CPU mode...
echo.
set CUDA_VISIBLE_DEVICES=
%PYTHON% detection_gui.py
goto end

:cuda_mode
echo.
echo Starting Hybrid Assistant in CUDA mode...
echo NOTE: Make sure NVIDIA CUDA toolkit is installed
echo.
set CUDA_VISIBLE_DEVICES=0
%PYTHON% detection_gui.py
goto end

:hybrid_mode
echo.
echo Starting Hybrid Assistant in HYBRID mode...
echo (Auto-detecting GPU if available)
echo.
REM Don't set CUDA_VISIBLE_DEVICES - let it auto-detect
%PYTHON% detection_gui.py
goto end

:exit_script
echo Exiting...
endlocal
exit /b 0

:end
echo.
echo Hybrid Assistant has closed.
pause
endlocal
exit /b 0
