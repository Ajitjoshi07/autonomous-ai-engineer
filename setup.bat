@echo off
REM ================================================================
REM  Autonomous AI Engineer — Fixed Windows Setup Script
REM  Works with Python 3.13 on Windows
REM  Run from: C:\Users\ajitm\OneDrive\Desktop\autonomous-ai-engineer
REM ================================================================

echo.
echo  ============================================
echo   AUTONOMOUS AI SOFTWARE ENGINEER - SETUP
echo   by Ajit Mukund Joshi
echo  ============================================
echo.

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH.
    pause & exit /b 1
)
echo [OK] Python found:
python --version

REM Python 3.13 on Windows sometimes breaks "pip" command
REM Always use "python -m pip" instead
echo.
echo Checking pip via python -m pip...
python -m pip --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo pip missing, installing via ensurepip...
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
)
echo [OK] pip ready:
python -m pip --version

echo.
echo Creating virtual environment...
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create venv & pause & exit /b 1
)
echo [OK] Virtual environment created

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip inside venv...
python -m pip install --upgrade pip setuptools wheel -q
echo [OK]

echo.
echo ============================================
echo Installing dependencies (5-10 minutes)...
echo ============================================

echo [1/7] python-dotenv, rich, pydantic...
python -m pip install python-dotenv rich pydantic -q && echo [OK] || echo [WARN] some failed

echo [2/7] LLM: openai, groq, langchain, langgraph...
python -m pip install openai groq langchain langchain-openai langgraph -q && echo [OK] || echo [WARN]

echo [3/7] Code understanding: faiss, sentence-transformers, networkx...
python -m pip install faiss-cpu sentence-transformers networkx gitpython -q && echo [OK] || echo [WARN]

echo [4/7] tree-sitter (AST parser)...
python -m pip install tree-sitter tree-sitter-python -q && echo [OK] || echo [WARN]

echo [5/7] GitHub + patch tools...
python -m pip install PyGithub unidiff docker -q && echo [OK] || echo [WARN]

echo [6/7] Test and quality tools...
python -m pip install pytest pytest-json-report ruff -q && echo [OK] || echo [WARN]

echo [7/7] FastAPI (optional dashboard)...
python -m pip install fastapi uvicorn -q && echo [OK] || echo [WARN]

echo.
echo Creating data directories...
if not exist "data\memory_store" mkdir "data\memory_store"
if not exist "data\faiss_index"  mkdir "data\faiss_index"
if not exist "data\logs"         mkdir "data\logs"
echo [OK] Directories ready

echo.
echo Setting up config\.env...
if not exist "config\.env" (
    if exist "config\.env.example" (
        copy "config\.env.example" "config\.env" >nul
        echo [OK] config\.env created from template
    ) else (
        (
            echo LLM_PROVIDER=groq
            echo LLM_MODEL=llama-3.1-70b-versatile
            echo LLM_MINI_MODEL=llama-3.1-8b-instant
            echo GROQ_API_KEY=PASTE_YOUR_GROQ_KEY_HERE
            echo GITHUB_TOKEN=PASTE_YOUR_GITHUB_TOKEN_HERE
            echo GITHUB_USERNAME=PASTE_YOUR_GITHUB_USERNAME_HERE
            echo SANDBOX_MEMORY_MB=512
            echo SANDBOX_TIMEOUT_SECONDS=30
            echo MAX_RETRY_ITERATIONS=8
            echo FAISS_INDEX_PATH=./data/faiss_index
            echo MEMORY_STORE_PATH=./data/memory_store
            echo LOGS_PATH=./data/logs
        ) > "config\.env"
        echo [OK] config\.env created - edit it with your API keys
    )
) else (
    echo [SKIP] config\.env already exists
)

echo.
echo ============================================
echo Verifying key packages...
echo ============================================
python -c "import rich; print('[OK] rich')"
python -c "import groq; print('[OK] groq')"
python -c "import openai; print('[OK] openai')"
python -c "import langgraph; print('[OK] langgraph')"
python -c "import faiss; print('[OK] faiss')"
python -c "import sentence_transformers; print('[OK] sentence_transformers')"
python -c "import pytest; print('[OK] pytest')"
python -c "import github; print('[OK] PyGithub')"

echo.
echo ============================================
echo  SETUP COMPLETE!
echo ============================================
echo.
echo NOW DO THIS:
echo.
echo   1. Open config\.env in Notepad
echo      Replace PASTE_YOUR_GROQ_KEY_HERE with your real key
echo      Replace PASTE_YOUR_GITHUB_TOKEN_HERE with your token
echo      Replace PASTE_YOUR_GITHUB_USERNAME_HERE with your username
echo.
echo   2. Run: python scripts\check_setup.py
echo.
echo   3. Run: python scripts\test_layers.py
echo.
pause
