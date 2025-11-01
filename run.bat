@echo off
REM Agentic Inventory Management System - Windows Setup and Run Script

echo 🤖 Agentic Inventory Management System
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Display Python version
echo Python version:
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models

:menu
echo.
echo Select an option:
echo 1. Run single inventory cycle
echo 2. Start automated scheduler
echo 3. Launch dashboard
echo 4. Train prediction models
echo 5. Interactive mode
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto run_cycle
if "%choice%"=="2" goto start_scheduler
if "%choice%"=="3" goto launch_dashboard
if "%choice%"=="4" goto train_models
if "%choice%"=="5" goto interactive_mode
if "%choice%"=="6" goto exit_script
echo Invalid option. Please select 1-6.
goto menu

:run_cycle
echo Running inventory management cycle...
python main.py cycle
goto continue

:start_scheduler
echo Starting automated scheduler...
echo Press Ctrl+C to stop
python main.py scheduler
goto continue

:launch_dashboard
echo Launching Streamlit dashboard...
echo Dashboard will open in your browser
python main.py dashboard
goto continue

:train_models
echo Training prediction models...
python main.py train
goto continue

:interactive_mode
echo Starting interactive mode...
python main.py
goto continue

:continue
echo.
pause
goto menu

:exit_script
echo Goodbye!
pause
exit /b 0
