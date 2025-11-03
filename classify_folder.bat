@echo off
REM Set UTF-8 encoding for emoji support
chcp 65001 > nul

REM Classify photos in a folder using Gemini AI
REM Usage: classify_folder.bat <input_folder_path>
REM Example: classify_folder.bat "C:\Photos\My Vacation"

if "%~1"=="" (
    echo Error: No input folder specified
    echo.
    echo Usage: classify_folder.bat ^<input_folder_path^>
    echo Example: classify_folder.bat "C:\Photos\My Vacation"
    echo.
    echo The script will create an output folder named: ^<input_folder^>_sorted
    echo with subfolders: Share, Storage, Review, Ignore, Videos
    pause
    exit /b 1
)

echo ============================================================
echo LLM Taste Cloner - Photo Classification
echo ============================================================
echo Input folder: %~1
echo.

REM Set Python to use UTF-8 encoding
set PYTHONIOENCODING=utf-8
python taste_classify_gemini_v4.py "%~1"

if errorlevel 1 (
    echo.
    echo Classification failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Classification complete!
echo Check the output folder: %~1_sorted
echo ============================================================
pause
