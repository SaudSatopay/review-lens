@echo off
rem ============================================================
rem  ReviewLens - one-click local launch
rem
rem  Double-click this file. On the first run it installs the
rem  core (no-torch) stack and the package itself, then starts
rem  the dashboard; every later run goes straight to launch.
rem  The browser opens at http://localhost:8501 automatically.
rem ============================================================
setlocal
title ReviewLens
pushd "%~dp0"

rem -- pick a Python: the project venv first, then whatever is on PATH
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [launch] Python was not found. Install Python 3.10+ from python.org
    echo          ^(tick "Add python.exe to PATH"^) and run this file again.
    pause
    exit /b 1
)

rem -- first run: install the core dependencies + ReviewLens itself
"%PY%" -c "import streamlit, reviewlens" >nul 2>&1
if errorlevel 1 (
    echo [launch] First run - installing the core stack ^(no torch, ~1 min^)...
    "%PY%" -m pip install -r requirements-core.txt || goto :fail
    "%PY%" -m pip install -e . || goto :fail
    echo [launch] Fetching NLTK data...
    "%PY%" -c "from reviewlens.nltk_setup import ensure_nltk_data; ensure_nltk_data()" || goto :fail
)

rem -- suppress Streamlit's first-run email prompt (it blocks one-click launch;
rem    a blank credentials file is exactly what answering it empty writes)
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    >  "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo [launch] Starting the ReviewLens journal - http://localhost:8501
echo [launch] Keep this window open; press Ctrl+C here to stop the app.
echo.
"%PY%" -m streamlit run app\streamlit_app.py
popd
exit /b 0

:fail
echo.
echo [launch] Setup failed - see the messages above, then run this file again.
pause
exit /b 1
