# Windows setup for smash-imagegen.
# Run from the repo root in PowerShell:
#   .\scripts\setup-windows.ps1
#
# Requires Python 3.11 already installed. Check with:
#   python --version
# If missing, install from python.org or:
#   winget install Python.Python.3.11

$ErrorActionPreference = "Stop"

Write-Host "==> Checking Python..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found. Install Python 3.11 first."
    exit 1
}

Write-Host "==> Creating virtual environment in .venv..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "==> Activating venv..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# PyTorch with CUDA 12.4 — must be installed before requirements.txt so we
# don't accidentally get the CPU-only build. Adjust cu124 -> cu121 if you
# have an older CUDA driver.
Write-Host "==> Installing PyTorch (CUDA 12.4)... this is the big download" -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "==> Installing remaining requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "==> Verifying CUDA is visible to PyTorch..." -ForegroundColor Cyan
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

Write-Host ""
Write-Host "==> Done. To run the server:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    python -m uvicorn server.main:app --host 0.0.0.0 --port 8000"
Write-Host ""
Write-Host "First request will download ~15-20GB of model weights and take a few minutes." -ForegroundColor Yellow
