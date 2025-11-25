@echo off
echo ======================================================
echo   Creating WhisperX Conda Environment
echo ======================================================

REM Change directory to the dialogue_extraction folder
cd Run_Dialogue_Extraction\whisperx_env.yml

REM Create the whisperx environment from the YAML file
conda env create -f whisperx_env.yml

echo.
echo ✅ WhisperX environment created successfully!
echo.
pause




