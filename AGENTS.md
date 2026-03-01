# Agent Instructions for jj-qfin-math

This file contains instructions for AI agents working on the jj-qfin-math project.

## Project Overview

- **Purpose**: Experimental and educational Python library for quantitative finance
- **Language**: Python 3.13+
- **Dependencies**: NumPy, SciPy
- **Testing**: pytest

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/bsm_test.py -v
```

## Code Style

- **Linting**: ruff (configured in pyproject.toml)
- **Type checking**: pyright (basic mode)
- **Docstrings**: numpy conventions
- **Comments**: Avoid adding inline (`#`) comments unless explicitly asked by the user.
  Keep existing user comments. Keep comments that contain instructions to linters
  and typecheckers (e.g., `# noqa:`, `# type: ignore`, `# noqa: D417`).

## Argument Ordering Convention

For all options-related functions, use this consistent argument order:

1. **Underlying data (price, vol)**: `s0`, `sigma`, `q` (dividend yield)
2. **Option data: ttm, strike, side**: `t`, `k`, `option_type`/`side`, `option_price` (if available)
3. **Market data**: `r` (risk-free rate), 
4. **Other technical args**: optimizer settings, bounds, etc.

### Example

```python
# bsm_option_eu_prices - correct order
call, put = bsm_option_eu_prices(
    s0=50.0,    # 1) Underlying: spot price
    sigma=0.3,  # 1) Underlying: volatility
    t=0.25,     # 2) Option: ttm
    k=50.0,     # 2) Option: strike
    r=0.05,     # 3) Market: risk-free rate
    q=0.0,      # 3) Market: dividend yield
)
```

## Mathematical Conventions

- Use naming: `s0` (spot), `sigma`/`vol` (volatility), `k` (strike), `t` (ttm), `r` (risk-free), `q` (dividend)
- Always specify units in docstrings (e.g., "in years", "annualized")
- Use float for scalar values, numpy arrays for vectors

## File Structure

```
src/jj_qfin_math/
  bsm.py      # Black-Scholes-Merton option pricing
  gbm.py      # Geometric Brownian Motion forecasting
tests/
  bsm_test.py
  gbm_test.py
```

## Key Patterns

- Use type hints for all function signatures
- Use `Final` for constants
