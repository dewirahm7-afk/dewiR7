@echo off
title Dewa Dracin - AutoDub (DEV MODE)
echo ========================================
echo    DEWA Dubbing - Starting (DEV MODE)...
echo ========================================
echo Environment: venv_clean
echo Python: 3.9.13
echo GPU: RTX 3050 ti 4GB - ELEK
echo ========================================

cd /d D:\xiaodub\dracindub_web
call D:\dubdracin\venv_clean\Scripts\activate.bat

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
