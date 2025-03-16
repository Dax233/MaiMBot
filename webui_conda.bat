@echo on
echo Starting script...
echo Activating conda environment: maimbot
call conda activate maimbot
if errorlevel 1 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)
echo Conda environment activated successfully
echo Changing directory to C:\GitHub\MaiMBot
cd /d C:\GitHub\MaiMBot
if errorlevel 1 (
    echo Failed to change directory
    pause
    exit /b 1
)
echo Current directory is:
cd

python webui.py
if errorlevel 1 (
    echo Command failed with error code %errorlevel%
    pause
    exit /b 1
)
echo Script completed successfully
pause