@echo off
setlocal
cd /d %~dp0
title ForgeX v3.2.0-open-source Installer
set PYTHONIOENCODING=utf-8

echo ================================================
echo   ForgeX v3.2.0-open-source - Open Source Runtime Installer
echo ================================================
echo.

set "PY_CMD="
where py >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python 3.10+ not found.
    pause
    exit /b 1
)

call %PY_CMD% launcher.py install %*
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
    echo.
    echo [ERROR] Runtime installation failed with exit code %ERR%.
)
pause
exit /b %ERR%
