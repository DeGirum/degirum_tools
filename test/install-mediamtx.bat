@echo off
setlocal

:: Script to install MediaMTX from GitHub releases on Windows

:: Set MediaMTX version
set MTX_VERSION=v1.12.3

:: Construct download URL (Windows amd64)
set FILENAME=mediamtx_%MTX_VERSION%_windows_amd64.zip
set URL=https://github.com/bluenviron/mediamtx/releases/download/%MTX_VERSION%/%FILENAME%

:: Install directory - use a folder inside LOCALAPPDATA so no admin rights are needed
set INSTALL_DIR=%LOCALAPPDATA%\mediamtx

echo Downloading %URL%
curl -fsSL -o "%TEMP%\%FILENAME%" "%URL%"
if errorlevel 1 (
    echo Failed to download MediaMTX from %URL%
    exit /b 1
)

:: Create install directory
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Extract zip
powershell -NoProfile -Command "Expand-Archive -Force -Path '%TEMP%\%FILENAME%' -DestinationPath '%INSTALL_DIR%'"
if errorlevel 1 (
    echo Failed to extract MediaMTX
    exit /b 1
)

:: Clean up downloaded archive
del /f /q "%TEMP%\%FILENAME%"

:: Add install directory to PATH for the current process and persist it for the user session
:: On a GH runner this sets it in GITHUB_PATH so subsequent steps pick it up
if defined GITHUB_PATH (
    echo %INSTALL_DIR%>> "%GITHUB_PATH%"
    echo Added %INSTALL_DIR% to GITHUB_PATH
) else (
    :: Persist for the current user (no admin required)
    for /f "usebackq tokens=2,*" %%A in (
        `reg query "HKCU\Environment" /v PATH 2^>nul`
    ) do set CURRENT_PATH=%%B
    if defined CURRENT_PATH (
        reg add "HKCU\Environment" /v PATH /t REG_EXPAND_SZ /d "%CURRENT_PATH%;%INSTALL_DIR%" /f >nul
    ) else (
        reg add "HKCU\Environment" /v PATH /t REG_EXPAND_SZ /d "%INSTALL_DIR%" /f >nul
    )
    echo Added %INSTALL_DIR% to user PATH
)

:: Make mediamtx available in the current shell session too
set PATH=%PATH%;%INSTALL_DIR%

echo MediaMTX %MTX_VERSION% installed to %INSTALL_DIR%
mediamtx --version

endlocal & set PATH=%PATH%;%INSTALL_DIR%
