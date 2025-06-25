@echo off
echo ========================================
echo Prometheus-L6 セットアップスクリプト
echo ========================================
echo.

:: Python の確認
python --version >nul 2>&1
if errorlevel 1 (
    echo [エラー] Python が見つかりません。
    echo Python 3.9以上をインストールしてください。
    pause
    exit /b 1
)

echo [OK] Python が見つかりました。

:: 仮想環境の作成
echo.
echo 仮想環境を作成中...
python -m venv venv
if errorlevel 1 (
    echo [エラー] 仮想環境の作成に失敗しました。
    pause
    exit /b 1
)

echo [OK] 仮想環境を作成しました。

:: 仮想環境の有効化
echo.
echo 仮想環境を有効化中...
call venv\Scripts\activate.bat

:: 依存パッケージのインストール
echo.
echo 依存パッケージをインストール中...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [エラー] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo [OK] パッケージのインストールが完了しました。

:: ディレクトリ構造の確認・作成
echo.
echo ディレクトリ構造を確認中...

if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "output\cache" mkdir output\cache
if not exist "output\db" mkdir output\db
if not exist "output\trained_models" mkdir output\trained_models

echo [OK] ディレクトリ構造を確認しました。

:: データファイルの確認
echo.
echo データファイルを確認中...

if not exist "data\LOTO6情報.csv" (
    echo [警告] data\LOTO6情報.csv が見つかりません。
    echo このファイルを手動で配置してください。
)

if exist "data\LOTO6情報.csv" (
    echo [OK] LOTO6情報.csv が見つかりました。
)

:: 設定完了
echo.
echo ========================================
echo セットアップが完了しました！
echo ========================================
echo.
echo 使用方法:
echo 1. data\LOTO6情報.csv にデータを配置
echo 2. start_app.bat を実行してアプリを起動
echo.
echo または手動で以下を実行:
echo venv\Scripts\activate
echo streamlit run main.py
echo.
pause
