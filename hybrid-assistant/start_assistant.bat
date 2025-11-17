@echo off
cd /d "%~dp0"

echo.
echo ============================================================
echo    HYBRID ASSISTANT - Mode Selector
echo ============================================================
echo.
echo Choose operation mode:
echo   [1] GPU mode (CUDA) - Fast but may crash
echo   [2] CPU mode - Stable (RECOMMENDED)
echo   [3] HYBRID mode - GPU with CPU fallback
echo.
set /p choice="Select [1/2/3]: "

if "%choice%"=="1" (
    set MODE=--gpu
    set MODE_NAME=GPU
) else if "%choice%"=="2" (
    set MODE=--cpu
    set MODE_NAME=CPU
) else if "%choice%"=="3" (
    set MODE=--hybrid
    set MODE_NAME=HYBRID
) else (
    echo Invalid choice. Using CPU mode...
    set MODE=--cpu
    set MODE_NAME=CPU
)

set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set LOGFILE=logs\assistant_%MODE_NAME%_%TIMESTAMP%.log

if not exist logs\ mkdir logs

echo.
echo Starting in %MODE_NAME% mode...
echo Logs will be saved to: %LOGFILE%
echo.

REM Run Python and save output to log file
.venv\Scripts\python.exe start_assistant.py %MODE% > %LOGFILE% 2>&1

echo.
echo ============================================================
echo    GUI Closed - Log saved to: %LOGFILE%
echo ============================================================
echo.
pause