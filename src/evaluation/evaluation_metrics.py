import polars as pl
import numpy as np
from plotnine import *

def compare_model_fits(rfm_test, predictions1, predictions2, model1_name="Model 1", model2_name="Model 2"):
    """
    Compare two model fits and visualize their performance against actual data.
    
    Parameters:
    -----------
    rfm_test : polars.DataFrame
        Test dataset containing 'x_test' and 'actual_holdout_spend' columns
    predictions1 : array-like
        Predictions from first model
    predictions2 : array-like
        Predictions from second model
    model1_name : str
        Name of the first model for legend
    model2_name : str
        Name of the second model for legend
    
    Returns:
    --------
    plotnine.ggplot
        Visualization comparing the two models
    """
    
    rfm_test = rfm_test.with_columns(
        x = pl.col('x').clip(upper_bound=8),
        actual_holdout_spend = pl.col('x_test') * pl.col('zbar_test'),
        expected_holdout_spend_1 = pl.col('x_test') *  np.array(predictions1),
        expected_holdout_spend_2 = pl.col('x_test') * np.array(predictions2)
    )
    
    total_spend_holdout = rfm_test\
        .group_by('x')\
        .agg(
            total_spend_by_x = pl.col('actual_holdout_spend').mean(),
            expected_total_spend_by_x_1 = pl.col('expected_holdout_spend_1').mean(),
            expected_total_spend_by_x_2 = pl.col('expected_holdout_spend_2').mean()
        )\
        .sort('x')
    
    # Create complete range of x values
    complete_range = pl.DataFrame({
        'x': range(rfm_test['x'].min(), rfm_test['x'].max() + 1)
    }).with_columns(pl.col('x').cast(pl.UInt32))
    
    # Calculate proportions
    props = rfm_test['x']\
        .value_counts(normalize=True)\
        .sort('x')\
        .join(complete_range, on='x', how='right')\
        .with_columns(pl.col('proportion').fill_null(0))
    
    # Create labels with proportions
    labels = [f"{count}\n({prop:.1%})" for count, prop in zip(props['x'], props['proportion'])]
    labels = [f"{props['x'][i]}+\n({props['proportion'][i]:.1%})" if i == len(labels)-1 
             else labels[i] for i in range(len(labels))]
    
    # Create plot
    p = (ggplot(total_spend_holdout) +
        geom_line(aes(x='x', y='total_spend_by_x', color='"Actual"'), size=2) +
        geom_line(aes(x='x', y='expected_total_spend_by_x_1', 
                 color=f'"{model1_name}"'), 
                 linetype='dashed', size=2) +
        geom_line(aes(x='x', y='expected_total_spend_by_x_2', 
                 color=f'"{model2_name}"'), 
                 linetype='dotted', size=2) +
        scale_x_continuous(
            breaks=range(1, int(total_spend_holdout['x'].max()) + 1),
            labels=labels
        ) +
        scale_y_continuous(limits=(0, None)) +
        scale_color_manual(
            values=['black', 'red', 'blue'],
            labels=['Total Spend', model1_name, model2_name]
        ) +
        labs(
            x='Frequency (in-sample)',
            y='Expected Total Spend in Holdout ($)',
            title='Conditional Expectations of Monetary Value',
            color=''
        ) +
        theme_minimal() +
        theme(
            panel_grid_major=element_line(),
            panel_grid_minor=element_blank(),
            legend_position='bottom',
            legend_box='horizontal',
            legend_title=element_text(size=10),
            legend_text=element_text(size=9)
        )
    )
    
    return p

def calculate_metrics(rfm_test, predictions1, predictions2, model1_name="Model 1", model2_name="Model 2"):
    """
    Calculate comparison metrics for two models and create a visualization.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predictions1 : array-like
        Predictions from first model
    predictions2 : array-like
        Predictions from second model
    model1_name : str
        Name of the first model
    model2_name : str
        Name of the second model
    
    Returns:
    --------
    plotnine.ggplot
        Visualization of metrics in table format
    """
    rfm_test = rfm_test.with_columns(
        x = pl.col('x').clip(upper_bound=8),
        actual_holdout_spend = pl.col('x_test') * pl.col('zbar_test'),
        expected_holdout_spend_1 = pl.col('x_test') *  np.array(predictions1),
        expected_holdout_spend_2 = pl.col('x_test') * np.array(predictions2)
    )
    
    actual = np.array(rfm_test['actual_holdout_spend'])
    predictions1 = np.array(rfm_test['expected_holdout_spend_1'])
    predictions2 = np.array(rfm_test['expected_holdout_spend_2'])
    
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate metrics
    metrics_data = {
        'Metric': ['MSE', 'MAE', 'MAPE (Total)'],
        model1_name: [
            mse(actual, predictions1),
            mae(actual, predictions1),
            mape(actual.sum(), predictions1.sum())
        ],
        model2_name: [
            mse(actual, predictions2),
            mae(actual, predictions2),
            mape(actual.sum(), predictions2.sum())
        ]
    }
    
    # Create Polars DataFrame
    df = pl.DataFrame(metrics_data)
    
    # Melt the DataFrame for ggplot
    df_melted = df.melt(
        id_vars=['Metric'],
        value_vars=[model1_name, model2_name],
        variable_name='variable',
        value_name='value'
    )
    
    # Format numbers and add as new column
    df_melted = df_melted.with_columns([
        pl.col('value').map_elements(lambda x: f"{x:,.2f}", return_dtype=pl.Utf8).alias('formatted_value')
    ])
    
    # Create the plot
    p = (ggplot() +
        geom_text(data=df_melted,
                 mapping=aes(x='variable', 
                            y='Metric',
                            label='formatted_value'),
                 size=8) +
        geom_tile(data=df_melted,
                  mapping=aes(x='variable',
                             y='Metric',
                             fill='value'),
                  alpha=0.1) +
        scale_fill_gradient2(low='blue', high='red', mid='white',
                           midpoint=df_melted['value'].mean()) +
        theme_minimal() +
        theme(
            axis_text_x=element_text(angle=0, hjust=0.5),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_position='none',
            plot_title=element_text(hjust=0.5)
        ) +
        labs(title='Model Comparison Metrics')
    )
    
    return p

# Example usage:
# p = calculate_metrics(actual, ghr, gg, "GHR Model", "GG Model")
# p.draw(show=True)
