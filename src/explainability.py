"""Model explainability using SHAP."""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_shap_explanations(
    model_path: Path = Path("main/model.pkl"),
    data_path: Path = Path("data/raw/data.csv"),
    output_dir: Path = Path("outputs/shap")
) -> None:
    """Generate SHAP explanations for the trained model.
    
    Parameters
    ----------
    model_path : Path
        Path to the trained model pickle file
    data_path : Path
        Path to the data CSV file
    output_dir : Path
        Directory to save SHAP plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic feature engineering
    from src.data_processing import engineer_features
    X, y = engineer_features(df)
    
    # Preprocess features (same as training)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    low_card_cols = [c for c in X.columns if X[c].dtype == "object" and X[c].nunique() <= 50]
    X_enc = pd.get_dummies(X[low_card_cols], drop_first=True)
    X = pd.concat([X[num_cols], X_enc], axis=1)
    X = X.select_dtypes(include=["number"])
    
    # Sample data for SHAP (use subset for speed)
    sample_size = min(100, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    logger.info(f"Generating SHAP explanations for {sample_size} samples")
    
    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}. Trying KernelExplainer...")
        # Fallback to KernelExplainer
        background = shap.sample(X, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    # 1. Summary Plot (Global Feature Importance)
    logger.info("Creating summary plot")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot - Global Feature Importance", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar Plot (Mean Absolute SHAP Values)
    logger.info("Creating bar plot")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance - Mean Absolute Impact", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall Plot (Individual Prediction Explanation)
    logger.info("Creating waterfall plot for first sample")
    plt.figure(figsize=(10, 8))
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
        data=X_sample.iloc[0].values,
        feature_names=X_sample.columns.tolist()
    )
    shap.waterfall_plot(shap_explanation, show=False)
    plt.title("SHAP Waterfall Plot - Individual Prediction", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Force Plot (save as HTML)
    logger.info("Creating force plot")
    try:
        force_plot = shap.force_plot(
            explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            shap_values[0],
            X_sample.iloc[0],
            matplotlib=False
        )
        shap.save_html(str(output_dir / "shap_force_plot.html"), force_plot)
    except Exception as e:
        logger.warning(f"Could not create force plot: {e}")
    
    # 5. Dependence Plot (for top feature)
    logger.info("Creating dependence plot")
    top_feature_idx = np.abs(shap_values).mean(axis=0).argmax()
    top_feature = X_sample.columns[top_feature_idx]
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature_idx,
        shap_values,
        X_sample,
        show=False
    )
    plt.title(f"SHAP Dependence Plot - {top_feature}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_dependence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP explanations saved to {output_dir}")
    
    # Generate summary report
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info("Feature importance saved to feature_importance.csv")
    
    return feature_importance


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument("--model-path", type=Path, default=Path("main/model.pkl"))
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shap"))
    
    args = parser.parse_args()
    
    generate_shap_explanations(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
