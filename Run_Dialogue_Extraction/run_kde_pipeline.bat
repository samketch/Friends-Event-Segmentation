@echo off
setlocal

REM ================================
REM KDE Pipeline: Clean -> Combine -> KDE -> Timeline
REM Runs from: Run_Dialogue_Extraction
REM ================================

REM Use UTF-8 so quotes/spaces are safe (optional; suppress output)
chcp 65001 >nul

REM Resolve repo root as the parent of this folder
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.."
set "REPO_ROOT=%CD%"

REM Key paths
set "ANALYSIS=%REPO_ROOT%\Analysis\event_seg analysis"
set "KERNAL=%ANALYSIS%\Analyzed_Data\Kernal"

echo --------------------------------------------------------
echo Repository root:        %REPO_ROOT%
echo Analysis folder:        %ANALYSIS%
echo Kernal output folder:   %KERNAL%
echo --------------------------------------------------------

REM Activate the psytask environment
call conda activate psytask
if errorlevel 1 (
  echo ERROR: Could not activate conda environment "psytask".
  goto :fail
)

REM ---- Step 1: Clean data ----
if not exist "%ANALYSIS%\clean_data.py" (
  echo ERROR: Missing "%ANALYSIS%\clean_data.py"
  goto :fail
)
echo [1/4] Cleaning data...
python "%ANALYSIS%\clean_data.py" || goto :fail

REM ---- Step 2: Combine CSVs ----
if not exist "%ANALYSIS%\combine_csv.py" (
  echo ERROR: Missing "%ANALYSIS%\combine_csv.py"
  goto :fail
)
echo [2/4] Combining CSV files...
python "%ANALYSIS%\combine_csv.py" || goto :fail

REM ---- Step 3: KDE analysis ----
if not exist "%KERNAL%\KDE.py" (
  echo ERROR: Missing "%KERNAL%\KDE.py"
  goto :fail
)
echo [3/4] Running KDE analysis...
pushd "%KERNAL%"
python "KDE.py" || (popd & goto :fail)

REM ---- Step 4: Timeline Graph (optional) ----
if exist "Timeline_graph.py" (
  echo [4/4] Generating Timeline Graph...
  python "Timeline_graph.py" || (popd & goto :fail)
) else (
  echo [4/4] Timeline_graph.py not found; skipping.
)
popd

echo.
echo SUCCESS: KDE pipeline completed. See outputs in:
echo   "%KERNAL%"
goto :end

:fail
echo.
echo FAILED: See messages above for details.
:end
popd
endlocal
pause
