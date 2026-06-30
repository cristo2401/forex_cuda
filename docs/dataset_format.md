# Dataset Format

The real-data preprocessing scripts expect one parquet file per FX pair:

```text
parquet_h1/
  EURUSD.parquet
  GBPUSD.parquet
  USDJPY.parquet
```

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `timestamp` | datetime-like | Bar timestamp. |
| `close` | float | Positive close price used for log returns. |

Optional columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `is_data_gap` | bool | Rows marked `True` are dropped when `drop_data_gaps=True`. |

The loader aligns all selected pairs by timestamp using an inner join, then computes:

```text
log_return[t] = log(close[t]) - log(close[t - 1])
```

Resulting tensor shape before rolling windows:

```text
[num_assets, total_steps]
```

Set the dataset path with:

```bash
export FOREX_CUDA_DATA_DIR=/absolute/path/to/parquet_h1
```

