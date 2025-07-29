@echo off
echo Starting Bear Detection Application...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if requirements are installed
python -c "import ultralytics" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo.
    echo Requirements installed.
    echo.
)

REM Run the application
echo Launching application...
echo.
python main.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)