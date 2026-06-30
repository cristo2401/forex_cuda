from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(close_prices: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    if close_prices.empty:
        raise ValueError("close_prices is empty")
    if (close_prices <= 0).any().any():
        raise ValueError("close_prices must be strictly positive")

    returns = np.log(close_prices).diff().dropna()
    if returns.empty:
        raise ValueError("not enough rows to compute log returns")

    log_returns = returns.to_numpy(dtype=np.float32).T.copy()
    return log_returns, pd.DatetimeIndex(returns.index), list(returns.columns)

