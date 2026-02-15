#!/bin/bash
# Bash script to run the Streamlit dashboard

echo "Starting Credit Risk Dashboard..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "main/model.pkl" ]; then
    echo "Model not found! Training model first..."
    python -m src.train --raw-path data/raw/data.csv --model-out main/model.pkl
fi

# Run Streamlit
echo "Launching dashboard at http://localhost:8501"
streamlit run src/dashboard.py
