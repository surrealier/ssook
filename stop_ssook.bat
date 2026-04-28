@echo off
taskkill /IM ssook.exe /F >nul 2>&1 && (
    echo ssook stopped.
) || (
    echo ssook is not running.
)
pause
