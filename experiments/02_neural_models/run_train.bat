@echo off
REM 使用 conda 环境 yh311_G 运行训练
REM 用法：在本目录下双击运行，或在命令行执行 run_train.bat
conda run -n yh311_G python "%~dp0train.py"
if errorlevel 1 (
    echo 若报错找不到 numpy/torch，请先在本目录打开终端并执行：
    echo   conda activate yh311_G
    echo   python train.py
)
pause
