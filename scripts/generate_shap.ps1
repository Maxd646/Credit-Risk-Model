# PowerShell script to generate SHAP explanations

Write-Host "Generating SHAP Explanations..." -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & venv\Scripts\Activate.ps1
}

# Check if model exists
if (-Not (Test-Path "main\model.pkl")) {
    Write-Host "Error: Model not found! Train the model first." -ForegroundColor Red
    exit 1
}

# Check if data exists
if (-Not (Test-Path "data\raw\data.csv")) {
    Write-Host "Error: Data file not found!" -ForegroundColor Red
    exit 1
}

# Generate SHAP explanations
Write-Host "Running SHAP analysis..." -ForegroundColor Cyan
python -m src.explainability --model-path main\model.pkl --data-path data\raw\data.csv --output-dir outputs\shap

Write-Host "SHAP visualizations saved to outputs\shap\" -ForegroundColor Green
