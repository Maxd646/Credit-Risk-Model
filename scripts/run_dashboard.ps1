# PowerShell script to run the Streamlit dashboard

Write-Host "Starting Credit Risk Dashboard..." -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & venv\Scripts\Activate.ps1
}

# Check if model exists
if (-Not (Test-Path "main\model.pkl")) {
    Write-Host "Model not found! Training model first..." -ForegroundColor Red
    python -m src.train --raw-path data\raw\data.csv --model-out main\model.pkl
}

# Run Streamlit
Write-Host "Launching dashboard at http://localhost:8501" -ForegroundColor Cyan
streamlit run src\dashboard.py
