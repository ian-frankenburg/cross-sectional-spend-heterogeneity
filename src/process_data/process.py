from datetime import datetime
import jax.numpy as jnp
import polars as pl

def process_txlog(txlog, T_max, min_spend=0.01):
    """
    Process transaction log data and create training/testing datasets with RFM metrics.
    
    Parameters:
    -----------
    txlog : polars.DataFrame
        Transaction log dataframe with columns: 'id', 'tx_time', 'spend'
    T_max : float
        Maximum time value for splitting data
    min_spend : float, default=0.01
        Minimum spend value to include in analysis
    
    Returns:
    --------
    tuple: (rfm_train, rfm_test)
        Two Polars DataFrames containing training and test data
    """
    # Filter and split data
    txlog_enriched = txlog.filter(pl.col('spend') >= min_spend)
    txlog_train = txlog_enriched.filter(pl.col("tx_time") <= T_max / 4)
    txlog_test = txlog_enriched.filter(pl.col("tx_time") > T_max / 4)
    
    # Create training RFM metrics
    rfm_train = txlog_train \
        .group_by('id') \
        .agg(
            x = pl.len(),
            zbar = pl.col('spend').mean(),
            zsum = pl.col('spend').sum(),
            zprod = pl.col('spend').product(),
        )\
        .sort('id')\
        .filter(pl.col('zsum') > 0)
    
    # Create testing RFM metrics
    rfm_test = txlog_test.group_by('id').agg(
        x_test = pl.len(),
        zbar_test = pl.col('spend').mean(),
        zsum_test = pl.col('spend').sum(),
        zprod_test = pl.col('spend').product(),
    ).sort('id').join(
        rfm_train,
        on='id',
        how='right'
    ).with_columns(
        x_test = pl.col('x_test').fill_null(0),
        zbar_test = pl.col('zbar_test').fill_null(0),
        zsum_test = pl.col('zsum_test').fill_null(0),
        zprod_test = pl.col('zprod_test').fill_null(0)
    ).select('id', 'x', 'zbar', 'x_test', 'zbar_test', 'zsum_test', 'zprod_test')
    
    return rfm_train, rfm_test

# Example usage:
# rfm_train, rfm_test = process_transaction_log(txlog, T_max)
# 
# zbar_train = rfm_train \
#     .select("zbar") \
#     .to_series() \
#     .to_list()
# zbar_train = jnp.array([np.array(sublist) for sublist in zbar_train])
# 
# zsum_train = rfm_train \
#     .select("zsum") \
#     .to_series() \
#     .to_list()
# zsum_train = jnp.array([np.array(sublist) for sublist in zsum_train])
# 
# zprod_train = rfm_train \
#     .select("zprod") \
#     .to_series() \
#     .to_list()
# zprod_train = jnp.array([np.array(sublist) for sublist in zprod_train])
# 
# x_train = rfm_train \
#     .select("x") \
#     .to_series() \
#     .to_list()
# x_train = jnp.array([np.array(sublist) for sublist in x_train])
