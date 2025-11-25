@echo off
setlocal

echo ======================================================
echo   Running Transcription + Semantic Analysis
echo ======================================================

REM Initialize conda base
call "C:\ProgramData\anaconda3\Scripts\activate.bat"

REM Activate the whisperx environment
call conda activate whisperx

REM Figure out repo root (one level up from this script’s folder)
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.."
set "REPO_ROOT=%CD%"
popd

REM Paths to python exe and scripts
set "PYEXE=C:\ProgramData\anaconda3\envs\whisperx\python.exe"
set "TRANSCRIPT=%REPO_ROOT%\Analysis\event_seg analysis\dialogue_extraction\extract_dialogue.py"
set "SEMANTICS=%REPO_ROOT%\Analysis\event_seg analysis\dialogue_extraction\semantics.py"

echo Repo root   : %REPO_ROOT%
echo Transcript  : %TRANSCRIPT%
echo Semantics   : %SEMANTICS%
echo ------------------------------------------------------

REM Check that files exist
if not exist "%TRANSCRIPT%" (
  echo ERROR: Missing file %TRANSCRIPT%
  pause
  goto :end
)
if not exist "%SEMANTICS%" (
  echo ERROR: Missing file %SEMANTICS%
  pause
  goto :end
)

REM Run transcription
echo.
echo 🎬 Running transcription...
"%PYEXE%" "%TRANSCRIPT%"

REM Run semantics
echo.
echo 🧠 Running semantic analysis...
"%PYEXE%" "%SEMANTICS%"

echo.
echo ✅ All done! Outputs saved in configured folders.
echo.
pause

:end
endlocal
