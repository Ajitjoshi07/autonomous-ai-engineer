@echo off
REM ================================================================
REM  Quick launcher — activates venv and opens a ready terminal
REM  Double-click this OR run from CMD every time you work on project
REM ================================================================

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo.
echo  ===============================================
echo   Autonomous AI Engineer - Environment Active
echo  ===============================================
echo.
echo  Commands available:
echo.
echo    python scripts\check_setup.py     Check API keys + dependencies
echo    python scripts\test_layers.py     Test each layer (green/yellow/red)
echo    python scripts\demo_run.py        Full demo (no API key needed)
echo    python -m pytest tests\ -v        Run all 24 unit tests
echo.
echo    python src\orchestrator.py --help  Run full pipeline
echo.
echo  Your project folder:
echo    C:\Users\ajitm\OneDrive\Desktop\autonomous-ai-engineer
echo.
cmd /k
