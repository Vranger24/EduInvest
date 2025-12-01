# EduInvest — Technical Documentation

This document explains the architecture and internals of the EduInvest application. It is intended for developers who want to understand or extend the project.

---

## Architecture overview

EduInvest is a single-process desktop application built in Python. It has three primary layers:

1. UI layer (PyQt6)
   - Provides windows, buttons, lists, and an embedded matplotlib chart.
2. Data layer (yfinance + helper functions)
   - Fetches historical price data for tickers and prepares inputs for visualization and prediction.
3. AI layer (placeholder / model wrapper)
   - A thin wrapper returns a direction prediction. The full model (conceptually) in `model.py` is a multimodal neural network combining Transformer, GNN, and LSTM components.

Communication between components is synchronous and happens on the main thread for simplicity (suitable for demo/hackathon). For production, consider moving network calls to a worker thread to keep the UI responsive.

---

## How the UI works

- `eduinvest.py` defines the `EduInvest` QMainWindow subclass.
- `initUI` builds the layout: left panel (score, controls), right panel (trending list and matplotlib canvas).
- `pickNewStock` (random stock picker) chooses a ticker and updates the label and chart.
- `handleGuess` compares the user's guess with the AI's direction and displays a result dialog.

Important UI elements:
- `QListWidget` for trending stocks
- `QPushButton` Up/Down for guesses
- `FigureCanvas` from `matplotlib.backends.backend_qtagg` shows historical close prices

---

## How the stock-fetching works

- `yfinance.Ticker(ticker).history(period=...)` is used to fetch OHLCV data.
- The app requests a default period (in the current implementation the chart uses `1y`), then converts the DataFrame index to `pandas.DatetimeIndex` and plots the `Close` column.
- Network errors or empty data are handled gracefully by showing a placeholder message on the chart.

Notes:
- yfinance uses Yahoo Finance endpoints and may be rate-limited. For robust usage consider caching results or providing an API key (if using a premium data source).

---

## How the AI prediction system plugs in

- The GUI calls `model_predict_direction(ticker)` which builds model inputs and calls the model's forward path.
- `build_model_inputs` collects transformer, GNN, and sentiment features. For the hackathon demo, the placeholder model returns a simple `Up`/`Down` label.
- The full model architecture concept (in `model.py`) is multimodal — Transformer for time-series, GNN for cross-stock relationships, and LSTM+attention for sentiment.

Integration tips:
- Keep the model as a small wrapper function that returns a JSON-like result object with keys such as `directions`, `movements`, `multistep`, `ranges`, `horizons`.
- If the model is heavy and you don't want to distribute it, host it remotely (REST) and let the app query it.

---

## Explanation of every important function

Note: some function names below (e.g., `predictMarket`) are conceptual — they map to how the demo code names and organizes logic. Use the docstrings and comments in `eduinvest.py` for exact mappings.

### predictMarket(ticker)

Purpose:
- High-level wrapper that prepares inputs and returns predictions for the requested ticker.

Inputs:
- `ticker` (string)

Outputs:
- A dictionary/object with keys: `directions`, `movements`, `multistep`, `ranges`, `horizons`.

How it would work:
1. Call `build_model_inputs` to prepare transformer, gnn, and sentiment tensors.
2. Run `model(...)` to obtain predictions.
3. Post-process outputs to convert tensors into numpy arrays and prices; return.

In the demo, `model_predict_direction` is a simplified analog — it returns a single direction.

---

### explainResult(prediction, features)

Purpose:
- Produces a human-readable explanation for the model's prediction.

Inputs:
- `prediction` (model raw output)
- `features` (optional: the inputs used by the model)

Outputs:
- Text explaining the key drivers (e.g., "Model expects Up because recent returns are positive and sentiment is positive").

How to implement:
- Use simple heuristics: look at the sign of short-term returns, sentiment polarity, and correlation signals from the GNN.
- Optionally use SHAP or integrated gradients to compute feature importances for ML explainability.

---

### randomStock()

Purpose:
- Return a random ticker from a curated list for the game.

Inputs:
- None

Outputs:
- `ticker` string

Implementation in demo:
- `random.choice([...])` picks from a fixed list like `['AAPL','TSLA',...]`.

---

### getHistory(ticker, period='1y')

Purpose:
- Fetch historical OHLCV data for `ticker` using `yfinance`.

Inputs:
- `ticker` (string), `period` (string)

Outputs:
- Pandas `DataFrame` with OHLCV and datetime index, or `None` / empty `DataFrame` on failure.

Example:
```python
import yfinance as yf

def getHistory(ticker, period='1y'):
    df = yf.Ticker(ticker).history(period=period)
    return df
```

---

### EduInvest.loadNewStock(self, ticker)

Purpose:
- Centralized method to update UI and internal state when a new stock is selected.

What it should do:
- Set `self.currentStock`.
- Update label text.
- Update chart (call `updateStockPlot` or similar).
- Query model and store AI prediction (optionally cached).

In the current demo this is spread across `pickNewStock` and `updateStockPlot`.

---

### EduInvest.handleGuess(self, studentGuess)

Purpose:
- Handle a user's guess action (Up/Down), compare it with AI prediction, update score and streak, and display a result.

Steps:
1. Read `self.aiDirection` (the AI's predicted direction).
2. Compare with `studentGuess`.
3. Update `self.score` and `self.streak`.
4. Show a `QMessageBox` with the result.
5. Trigger selection of a new stock (call `pickNewStock` or `loadNewStock`).

---

## Flow diagrams / pseudo-diagrams (round system)

High-level flow per round (user action):

1. UI displays a stock and chart.
2. AI prediction is precomputed or fetched: `ai_direction = model_predict_direction(ticker)`.
3. User presses Up or Down.
4. `handleGuess` compares and updates scores.
5. Display result dialog.
6. Select new stock and update UI.

Pseudo-diagram (textual):

```
[Start] -> [Pick stock] -> [Fetch history] -> [Display chart] -> [Get AI prediction]
   -> [User guesses] -> [Compare guess vs AI] -> [Update score] -> [Show result]
   -> [Pick stock (loop)]
```

---

## Developer notes & suggestions

- Move network and heavy model calls off the main thread (use `QThread` or `concurrent.futures`) to keep UI responsive.
- Use caching for yfinance results (simple disk cache) to avoid rate-limits.
- Provide a `--headless` or `--server` mode to run the model-only logic for automated testing.

---

## Where to add code (file map)

- `eduinvest.py` — GUI and glue code (main window)
- `model.py` — multimodal model architectures and model wrappers (Transformer, GNN, sentiment LSTM)

---

If you want, I can also annotate the code with inline comments (docstrings and block comments) to make the functions clearer; tell me which files to annotate and I will insert comments without changing logic.