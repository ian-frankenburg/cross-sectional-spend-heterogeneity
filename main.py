"""
Example script demonstrating monetary value modeling pipeline
using the GG/GHR/GHSR models.
"""
import logging
from pathlib import Path

import plotnine as p9

from src.simulation import simulate_orders, simulate_spend
from src.process_data import process
from src.inference_forecast import inference, forecasting
from src.evaluation import evaluation_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simulation_pipeline(model_type="gg"):
    """
    Run the full simulation pipeline for a given model type.
    
    Args:
        model_type: One of ["gg", "ghr", "ghsr"]
        
    Returns:
        Tuple containing training data, test data, and model predictions
    """
    logger.info(f"Running simulation pipeline for {model_type.upper()} model")
    
    # Generate transaction data
    txlog = simulate_orders.run_simulation()
    
    # Apply spend model
    spend_functions = {
        "gg": simulate_spend.generate_gg_spends,
        "ghr": simulate_spend.generate_ghr_spends,
        "ghsr": simulate_spend.generate_ghsr_spends
    }
    txlog_enriched = spend_functions[model_type](txlog)
    
    # Process into RFM format
    train_data, test_data = process.process_txlog(
        txlog_enriched, 
        T_max=52 * 3  # 3 years
    )
    
    # Run inference
    results = inference.estimate_model(model_type, train_data)
    
    # Generate predictions
    predictions = forecasting.forecast(
        model_type,
        results['logparameters'],
        train_data
    )
    
    return train_data, test_data, predictions

def evaluate_models(test_data, predictions1, predictions2, model1_name, model2_name):
    """
    Evaluate and compare two models' predictions.
    """
    logger.info(f"Evaluating {model1_name} vs {model2_name}")
    
    # Generate comparison plots
    fit_plot = evaluation_metrics.compare_model_fits(
        test_data, predictions1, predictions2,
        model1_name, model2_name
    )
    fit_plot.draw(show=True)
    
    metrics_plot = evaluation_metrics.calculate_metrics(
        test_data, predictions1, predictions2,
        model1_name, model2_name
    )
    metrics_plot.draw(show=True)

def main():
    # Run pipeline for GG model
    train_gg, test_gg, pred_gg = run_simulation_pipeline("gg")
    
    # Optionally run other models
    # train_ghr, test_ghr, pred_ghr = run_simulation_pipeline("ghr")
    # train_ghsr, test_ghsr, pred_ghsr = run_simulation_pipeline("ghsr")
    
    # Compare models (currently comparing GG against itself as example)
    evaluate_models(test_gg, pred_gg, pred_gg, "GG Model", "GG Model")

if __name__ == "__main__":
    main()
