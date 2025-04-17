@echo off
echo Starting AI-based IDS System...

REM Request admin privileges
echo Requesting administrator privileges...
powershell Start-Process cmd -Verb RunAs -ArgumentList '/c cd /d "%~dp0" && venv311\Scripts\python.exe scripts/run_ids.py'

REM Start the UI
echo Starting IDS UI...
timeout /t 2 /nobreak
call venv311\Scripts\activate.bat
python ui\main.py

pause
