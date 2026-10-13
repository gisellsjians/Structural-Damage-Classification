@echo off
setlocal
cd /d "%~dp0"

echo ==== Setup environment: structural-damage-classification ====

if not exist venv (
    echo [1/4] Membuat virtual environment "venv"...
    py -m venv venv || goto :error
) else (
    echo [1/4] Folder venv sudah ada, dilewati.
)

echo [2/4] Upgrade pip...
venv\Scripts\python.exe -m pip install --upgrade pip || goto :error

echo [3/4] Install semua package dari requirements.txt...
venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error

echo [4/4] Daftarkan kernel Jupyter "Python (venv)"...
venv\Scripts\python.exe -m ipykernel install --user --name structural-damage --display-name "Python (venv)" || goto :error

echo.
echo Selesai. Buka notebook dan pilih kernel "Python (venv)".
pause
exit /b 0

:error
echo.
echo Terjadi kesalahan pada langkah di atas. Periksa pesan kesalahan yang ditampilkan.
pause
exit /b 1