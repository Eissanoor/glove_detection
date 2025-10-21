@echo off
echo Installing PyTorch and dependencies with DLL fix...

REM Create virtual environment
python -m venv glove_env
call glove_env\Scripts\activate.bat

REM Install PyTorch CPU version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
pip install ultralytics==8.0.196 opencv-python numpy pillow streamlit pyyaml

echo Installation complete!
echo To activate the environment, run: glove_env\Scripts\activate.bat
pause
