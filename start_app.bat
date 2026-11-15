@echo off
echo ========================================================
echo  Starting NASA C-MAPSS RUL Prediction Platform
echo ========================================================

start "NASA RUL Backend (FastAPI)" cmd /k "cd /d %~dp0backend && python main.py"
timeout /t 3 /nobreak >nul
start "NASA RUL Frontend (Vite/React)" cmd /k "cd /d %~dp0 && npm run dev"

echo.
echo Servers started:
echo   - Backend API: http://127.0.0.1:8000
echo   - Frontend UI:  http://localhost:3000
echo ========================================================
