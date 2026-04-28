@echo off
taskkill /IM ssook.exe /T /F >nul 2>&1 && (
    echo ssook stopped.
) || (
    echo ssook is not running.
)
pause
