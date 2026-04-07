@echo off
echo Stopping ssook...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do taskkill /pid %%a /f >nul 2>&1
wmic process where "commandline like '%%run_web.py%%' and name='python.exe'" call terminate >nul 2>&1
echo Done.
