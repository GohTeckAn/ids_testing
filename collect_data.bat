@echo off
echo Starting network traffic collection...
call venv311\Scripts\activate.bat
python network_data_collector.py
pause
