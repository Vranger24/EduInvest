# Contributing to EduInvest

Welcome — contributions are welcome! This guide explains how to extend the app, replace the placeholder AI, add UI pages, and more.

## Development setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install PyTorch and PyG as described in `README.md`.

4. Run the app locally:

```bash
python eduinvest.py
```

---

## How to extend the model

- The placeholder model is a simple wrapper. To add a real model:
  1. Create a `models/` folder and add your PyTorch model file (e.g., `my_model.py`) and saved weights (e.g., `my_model.pth`).
  2. Replace the wrapper `model_predict_direction` to load your weights and run inference.
  3. Ensure the input feature shapes match what the model expects.

- Testing tip: add a small `demo_weights.pth` that the app can load for demo inference.

---

## How to replace the placeholder AI with a real model

1. Implement a `ModelWrapper` class that:
   - Loads model architecture and weights.
   - Exposes a method `predict(inputs) -> dict` that returns the model outputs in a stable format.

2. Update `model_predict_direction` to use `ModelWrapper.predict` and extract the Up/Down choice.

3. Make sure long-running inference runs off the main thread to avoid blocking the UI.

---

## How to add new UI pages

- The UI is built using `QMainWindow` and standard Qt widgets.
- Typical steps:
  1. Create a new `QWidget` subclass for the page.
  2. Add it to the main layout or use a `QStackedWidget` to switch between pages.
  3. Add navigation controls (buttons or menu items) to switch pages.

---

## How to add more stock tickers

- The trending and random stocks are defined in arrays inside `eduinvest.py`.
- Add tickers as ticker symbols (e.g., `AAPL`, `TSLA`, `BTC-USD`) — ensure yfinance supports them.
- If you add many tickers, consider lazy-loading the history only when a ticker is selected to reduce startup time.

---

## Coding guidelines

- Keep changes small and isolated.
- Add docstrings for new functions and classes.
- Write unit tests for non-UI logic if possible.
- For UI changes, include screenshots or animated GIFs demonstrating the new behavior.

---

## Reporting issues & PR process

- Open an issue describing the bug or enhancement with steps to reproduce.
- Submit a pull request with a clear description, and reference the issue.
- Include screenshots and tests where relevant.

Thanks for contributing — this project is meant for rapid prototyping and learning, so small, incremental improvements are ideal.