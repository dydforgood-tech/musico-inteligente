@echo off
title Virtual Band AI - v0.2 - Baixista Virtual
chcp 65001 >nul

cd /d "%~dp0"

echo ===================================================
echo           VIRTUAL BAND AI - INICIALIZADOR
echo ===================================================
echo.

set "PYTHON_CMD="

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Utilizando ambiente virtual .venv...
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_CMD=python"
    ) else (
        where py >nul 2>nul
        if %ERRORLEVEL% EQU 0 (
            set "PYTHON_CMD=py"
        ) else (
            if exist "%LocalAppData%\Programs\Python\Python314\python.exe" (
                set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python314\python.exe"
            )
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERRO] Interpretador Python nao encontrado no sistema.
    echo Por favor instale o Python ou adicione-o ao PATH.
    pause
    exit /b 1
)

echo [INFO] Utilizando: %PYTHON_CMD%
echo [INFO] Abrindo janela desktop do Virtual Band AI...
echo.

"%PYTHON_CMD%" main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O aplicativo encerrou com o codigo de saida: %ERRORLEVEL%
    echo Pressione qualquer tecla para fechar esta janela.
    pause
)
