@echo off
setlocal

REM ================================
REM Permutation Test Runner (Corrected Path)
REM ================================

chcp 65001 >nul

REM Resolve repo root as parent of this script’s folder
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.."
set "REPO_ROOT=%CD%"
popd

REM Key paths
set "ANALYSIS=%REPO_ROOT%\Analysis\event_seg analysis"
set "PERM_DIR=%ANALYSIS%\Analyzed_Data\Semantics"
set "PERM_FILE=%PERM_DIR%\Perm.py"
set "PYEXE=C:\ProgramData\anaconda3\envs\psytask\python.exe"

echo ======================================================
echo   Running Permutation Test (Event Segmentation Analysis)
echo ======================================================
echo Repo root: %REPO_ROOT%
echo PERM_DIR : %PERM_DIR%
echo PERM_FILE: %PERM_FILE%
echo Python   : %PYEXE%
echo ------------------------------------------------------

if not exist "%PERM_FILE%" (
  echo ERROR: Missing file: "%PERM_FILE%"
  pause
  goto :fail
)

REM Initialize conda
call "C:\ProgramData\anaconda3\Scripts\activate.bat" >nul 2>&1

echo.
echo 🔄 Running permutation analysis...
pushd "%PERM_DIR%"
"%PYEXE%" "perm.py"
echo === DEBUG: Finished running perm.py ===
popd

echo.
echo ✅ Permutation test completed! Results saved in: %PERM_DIR%
goto :end

:fail
echo.
echo ❌ FAILED. See messages above.

:end
pause
endlocal
