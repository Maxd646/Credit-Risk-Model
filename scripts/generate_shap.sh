#!/bin/bash
# Bash script to generate SHAP explanations

echo "Generating SHAP Explanations..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "main/model.pkl" ]; then
    echo "Error: Model not found! Train the model first."
    exit 1
fi

# Check if data exists
if [ ! -f "data/raw/data.csv" ]; then
    echo "Error: Data file not found!"
    exit 1
fi

# Generate SHAP explanations
echo "Running SHAP analysis..."
python -m src.explainability --model-path main/model.pkl --data-path data/raw/data.csv --output-dir outputs/shap

echo "SHAP visualizations saved to outputs/shap/"
