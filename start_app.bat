@echo off
echo Prometheus-L6 を起動しています...
echo.

:: 仮想環境の有効化
call venv\Scripts\activate.bat

:: Streamlit アプリの起動
echo ブラウザでアプリが開きます...
echo 終了するには Ctrl+C を押してください。
echo.

streamlit run main.py

pause
