# jj_qfin_math

**Experimental** and **educational** Python library for quantitative and theoretical finance.

This project is a **work in progress**.

The API is unstable and subject to frequent breaking changes as it is primarily used for
internal research and development.

## 📌 Overview

`jj_qfin_math` provides a collection of tools for modeling financial markets.


## 🛠 Tech Stack

* **Python 3.13+**
* **SciPy/NumPy**
* **Pandas**


## Installation

For now package is not published to index, however it can be installed from source.

For example with `uv` it can be done in `pyproject.toml` file like this:

```toml
[project]
...
dependencies = ["jj-qfin-math"]

[tool.uv.sources]
jj-qfin-math = { git = "https://github.com/ihorh/jj-qfin-math" }
```


## 🚀 Example: GBM Forecasting

```python
from jj_qfin_math.gbm import gbm_stock_price_forecast

# Forecast a stock price 1 year out
forecast = gbm_stock_price_forecast(
    s0=100.0,
    t=1.0,
    mu=0.15,      # Arithmetic drift (Expected Return)
    sigma=0.20    # Volatility
)

print(f"Expected Price: {forecast.price_expected:.2f}")
print(f"Lower 95% Bound: {forecast.price_dist.interval(0.95)[0]:.2f}")
```


## Credits

Inspired by

* John Hull's *Options, Futures, and Other Derivatives*
* Daniel P. Palomar *Portfolio Optimization*


## ⚠️ Disclaimer

* This is a **theoretical** and **educational** tool.
* It is provided "as is" without warranty.
* For educational purposes only. This content is not financial or legal advice.


## 📝 License

&copy; 2026 Ihor H.

- **Code**: MIT License  
- **Article / Text**: Creative Commons Attribution 4.0 International (CC BY 4.0)  
