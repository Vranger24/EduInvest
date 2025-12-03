import sys
import random
import yfinance as yf
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

import torch
import signal
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd


from model import (
    MultimodalStockPredictor,
    create_transformer_features,
    create_gnn_features,
    create_sentiment_features,
    NewsAnalyzer
)

# ---------------------------------------------------------------------------
# Module overview
# ---------------------------------------------------------------------------
# This file contains the PyQt6 GUI and light glue logic for the EduInvest
# hackathon project. It provides:
# - `build_model_inputs`: prepares model input tensors from yfinance history.
# - `model_predict_direction`: lightweight wrapper that returns an Up/Down
#   prediction from the demo model.
# - `EduInvest`: the main QMainWindow subclass which builds the UI, displays
#   a price chart, and handles user guesses.
#
# The code intentionally keeps network and model calls synchronous for the
# prototype. For production use, move those calls to background threads.
# ---------------------------------------------------------------------------


MODEL_PATH = "multimodal_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORMER_FEATURES = 8
GNN_FEATURES = 5
SENTIMENT_FEATURES = 5

model = MultimodalStockPredictor(
    transformer_features=TRANSFORMER_FEATURES,
    gnn_features=GNN_FEATURES,
    sentiment_features=SENTIMENT_FEATURES
).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("AI Model Loaded Successfully.")
except Exception as e:
    print("⚠ Could not load model:", e)

model.eval()

# Track tickers we've warned about to avoid repeated logs
WARNED_TICKERS = set()
def build_model_inputs(ticker):
    """Prepare and return tensors required by the model for `ticker`.

    Returns a tuple: (trans_seq, gnn_x, edge_index, sent_x, sentiment_meta)

    - `trans_seq`: FloatTensor shaped for the transformer input (batch, seq, features)
    - `gnn_x`: FloatTensor with GNN node features
    - `edge_index`: LongTensor edge index for the GNN
    - `sent_x`: FloatTensor for sentiment/time features
    - `sentiment_meta`: raw sentiment values (kept for debug / explainability)

    Note: This function uses `yfinance` and the feature creation helpers from
    `model.py`. It does not run the model itself.
    """

    data = yf.Ticker(ticker).history(period="2y")

    X_trans, df = create_transformer_features(data)
    if len(X_trans) < 256:
        raise ValueError("Not enough price data for AI model.")

    seq = torch.tensor(X_trans[-256:], dtype=torch.float32).unsqueeze(0)

    closes = df["Close"].values
    volumes = df["Volume"].values

    gnn_features = create_gnn_features([closes], [volumes])
    gnn_x = torch.tensor(gnn_features, dtype=torch.float32)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    analyzer = NewsAnalyzer()
    sentiment = analyzer.get_stock_sentiment(
        ticker,
        prices=closes,
        volumes=volumes
    )

    sent_features = create_sentiment_features([closes], [volumes], [sentiment], seq_len=20)
    sent_x = torch.tensor(sent_features, dtype=torch.float32)

    return seq.to(device), gnn_x.to(device), edge_index.to(device), sent_x.to(device), sentiment


def model_predict_direction(ticker):
    """Return a simple Up/Down prediction (and confidence) for `ticker`.

    This wrapper builds model inputs with `build_model_inputs`, runs the
    (possibly heavy) model, and converts the direction logits into a class
    label and a scalar confidence score in [0,1].

    For the hackathon demo the returned confidence is approximate. In a
    production setting, you may want to calibrate the probabilities.
    """
    try:
        tx, gx, edge, sx, sent = build_model_inputs(ticker)

        outputs = model(tx, gx, edge, sx, target_idx=0)
        logits = outputs["directions"][0, 0]

        probs = torch.softmax(logits, dim=-1)
        down_p, up_p = probs.tolist()

        if up_p > down_p:
            return "Up", float(up_p)
        else:
            return "Down", float(down_p)

    except Exception as e:
        if ticker not in WARNED_TICKERS:
            print("Model error:", e)
            WARNED_TICKERS.add(ticker)
        return random.choice(["Up", "Down"]), 0.5



class EduInvest(QMainWindow):
        """Main window for the EduInvest application.

        Responsibilities:
        - Build and arrange UI widgets
        - Display a stock's recent history using a matplotlib `FigureCanvas`
        - Ask the AI for a direction prediction and compare it with the user's
            guess
        - Maintain simple scoring and streak state for the game
        """
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EduInvest - AI Stock Guessing Game")
            self.setGeometry(200, 200, 900, 600)

            self.score = 0
            self.streak = 0
            self.currentStock = None

            self.initUI()
            self.pickNewStock()

   
    def initUI(self):
        """Initialize and layout GUI widgets.

        Left panel: title, current stock label, score, streak, Up/Down buttons
        Right panel: trending stocks list, embedded matplotlib chart
        """
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)

        layout = QHBoxLayout()
        mainWidget.setLayout(layout)

        leftPanel = QVBoxLayout()

        self.title = QLabel("EduInvest 📈")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.stockLabel = QLabel("Loading...")
        self.stockLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stockLabel.setStyleSheet("font-size: 22px; margin-top: 20px;")

        self.scoreLabel = QLabel("Score: 0")
        self.scoreLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scoreLabel.setStyleSheet("font-size: 18px;")

        self.streakLabel = QLabel("Streak: 0🔥")
        self.streakLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.streakLabel.setStyleSheet("font-size: 18px; margin-bottom: 10px;")

        self.btnUp = QPushButton("📈 Up")
        self.btnDown = QPushButton("📉 Down")

        for btn in (self.btnUp, self.btnDown):
            btn.setFixedHeight(50)
            btn.setStyleSheet("font-size: 18px;")

        self.btnUp.clicked.connect(lambda: self.handleGuess("Up"))
        self.btnDown.clicked.connect(lambda: self.handleGuess("Down"))

        leftPanel.addWidget(self.title)
        leftPanel.addWidget(self.stockLabel)
        leftPanel.addWidget(self.scoreLabel)
        leftPanel.addWidget(self.streakLabel)
        leftPanel.addWidget(self.btnUp)
        leftPanel.addWidget(self.btnDown)

        layout.addLayout(leftPanel, stretch=2)

        rightPanel = QVBoxLayout()

        trendingLabel = QLabel("🔥 Trending Stocks")
        trendingLabel.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        trendingLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.trendingList = QListWidget()
        self.loadTrendingStocks()

        # Matplotlib canvas for stock price chart
        self.canvas = FigureCanvas(plt.Figure(figsize=(5, 3)))
        self.ax = self.canvas.figure.subplots()
        rightPanel.addWidget(self.canvas)

        rightPanel.addWidget(trendingLabel)
        rightPanel.addWidget(self.trendingList)

        layout.addLayout(rightPanel, stretch=1)

    def loadTrendingStocks(self):
        trending = ["AAPL", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "MSFT"]
        for t in trending:
            self.trendingList.addItem(t)

    def pickNewStock(self):
        """Pick a new random ticker for the next round and update UI.

        Right now this is synchronous and will call the model and fetch history
        on the UI thread. For responsiveness, move heavy work to a worker.
        """
        stocks = ["AAPL", "TSLA", "MSFT", "AMZN", "NVDA", "GOOGL", "META"]
        self.currentStock = random.choice(stocks)
        self.stockLabel.setText(f"Will {self.currentStock} go Up or Down tomorrow?")

        # Only keep direction; we no longer display confidence in the UI
        self.aiDirection, _ = model_predict_direction(self.currentStock)

        # Update the embedded stock chart whenever a new stock is picked
        try:
            self.updateStockPlot(self.currentStock)
        except Exception as e:
            print("Failed to update stock plot:", e)

    def handleGuess(self, studentGuess):
        """Process the player's guess and show the result dialog.

        Updates score and streak, then selects a new stock for the next round.
        """
        aiGuess = self.aiDirection
        if studentGuess == aiGuess:
            # Correct guess
            self.score += 10
            self.streak += 1
            result = f"✅ Correct!\n\nAI predicted: {aiGuess}"
        else:
            # Incorrect guess — reset streak
            self.streak = 0
            result = f"❌ Incorrect.\n\nAI predicted: {aiGuess}"

        self.scoreLabel.setText(f"Score: {self.score}")
        self.streakLabel.setText(f"Streak: {self.streak}🔥")

        QMessageBox.information(self, "Result", result)

        self.pickNewStock()


    def updateStockPlot(self, ticker, period="1y"):
        """Fetch recent history for `ticker` and draw a close-price chart on the canvas."""
        try:
            df = yf.Ticker(ticker).history(period=period)
            self.ax.clear()

            if df is None or df.empty:
                self.ax.text(0.5, 0.5, "No price data available", ha='center', va='center')
                self.ax.set_title(f"{ticker} - no data")
            else:
                dates = pd.to_datetime(df.index)
                closes = df['Close']
                self.ax.plot(dates, closes, color='tab:blue', linewidth=1.5)
                self.ax.set_title(f"{ticker} — Close Price ({period})")
                self.ax.set_ylabel('Close ($)')
                self.ax.grid(True, alpha=0.3)
                self.canvas.figure.autofmt_xdate()

            self.canvas.draw()
        except Exception as e:
            print("Plotting error:", e)



if __name__ == "__main__":
    # Ignore Ctrl+C/SIGINT so accidental terminal interrupts don’t close the UI
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    app = QApplication(sys.argv)
    window = EduInvest()
    window.show()
    sys.exit(app.exec())

