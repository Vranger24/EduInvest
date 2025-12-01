import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import warnings
from tqdm import tqdm
import sys
warnings.filterwarnings('ignore')

# ============================================================================
# COMPONENT MODELS
# ============================================================================

# 1. TRANSFORMER MODEL 
class GPTPrice(nn.Module):
    """
    Enhanced Transformer for quantitative trading
    
    OPTIMIZATION PARAMETERS:
    - d_model: Hidden dimension (default 128, try: 256, 512 for more capacity)
    - nhead: Number of attention heads (default 8, try: 16, 32)
    - num_layers: Transformer depth (default 4, try: 6, 8, 12)
    - seq_len: Sequence length (default 256, try: 512, 1024)
    - dropout: Regularization (default 0.1, range: 0.05-0.3)
    """
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4, seq_len=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.max_seq_len = max(seq_len, 256)
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,  # Increased from 2x to 4x
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Deeper output projection
        self.fc_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4, d_model//8)
        )
    
    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask)
        x = x[:, -1, :]
        return self.fc_out(x)


# 2. GNN MODEL
class StockGNN(nn.Module):
    """
    Enhanced Graph Neural Network for stock relationships
    
    OPTIMIZATION PARAMETERS:
    - hidden_dim: Node embedding size (default 128, try: 256, 512)
    - num_gat_layers: Graph conv depth (default 4, try: 6, 8)
    - heads: Attention heads per layer (default 8, try: 16)
    - dropout: Regularization (default 0.2, range: 0.1-0.4)
    """
    def __init__(self, input_dim, hidden_dim=128, num_gat_layers=4, heads=8, dropout=0.2):
        super(StockGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Middle layers
        for _ in range(num_gat_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Last layer (single head)
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        x = F.elu(x)
        x = self.fc(x)
        
        return x


# 3. SENTIMENT MODEL
class SentimentLSTM(nn.Module):
    """
    Enhanced LSTM for sentiment and temporal features
    
    OPTIMIZATION PARAMETERS:
    - hidden_dim: LSTM hidden size (default 128, try: 256, 512)
    - num_layers: LSTM depth (default 3, try: 4, 6)
    - num_heads: Attention heads (default 8, try: 16)
    - dropout: Regularization (default 0.2, range: 0.1-0.4)
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=8, dropout=0.2):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        features = attn_out[:, -1, :]
        return self.fc(features)


# ============================================================================
# MULTIMODAL FUSION ARCHITECTURE
# ============================================================================

class MultimodalStockPredictor(nn.Module):
    """
    Quant-level multimodal architecture
    
    OPTIMIZATION PARAMETERS:
    - hidden_dim: Fusion layer size (default 256, try: 512, 1024)
    - num_fusion_layers: Fusion depth (default 3, try: 4, 6)
    - dropout: Regularization (default 0.15, range: 0.1-0.3)
    """
    def __init__(self, transformer_features, gnn_features, sentiment_features, 
                 hidden_dim=256, num_fusion_layers=3, seq_len=256, dropout=0.15):
        super(MultimodalStockPredictor, self).__init__()
        
        # Enhanced individual models
        self.transformer = GPTPrice(transformer_features, d_model=128, nhead=8, num_layers=4, seq_len=seq_len)
        self.gnn = StockGNN(gnn_features, hidden_dim=128, num_gat_layers=4)
        self.sentiment = SentimentLSTM(sentiment_features, hidden_dim=128, num_layers=3)
        
        self._fusion_input_dim = None
        self.fusion = None
        self._hidden_dim = hidden_dim
        self._num_fusion_layers = num_fusion_layers
        self._dropout = dropout
        
        # Enhanced prediction heads with residual connections
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.multistep_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10)
        )
        
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10)
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 64),
            nn.GELU(),
            nn.Linear(64, 10 * 2)
        )
        
        self.range_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 64),
            nn.GELU(),
            nn.Linear(64, 10 * 2)
        )
        
        self.horizon_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 3)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, trans_x, gnn_x, gnn_edge_index, sent_x, target_idx=0):
        trans_feat = self.transformer(trans_x)
        gnn_all = self.gnn(gnn_x, gnn_edge_index)
        sent_feat = self.sentiment(sent_x)

        if isinstance(target_idx, torch.Tensor):
            target_idx = int(target_idx.item())
        batch_size = trans_feat.size(0)

        if target_idx >= gnn_all.size(0):
            raise ValueError(f"target_idx {target_idx} >= number of stocks {gnn_all.size(0)}")
        gnn_feat = gnn_all[target_idx].unsqueeze(0).repeat(batch_size, 1)

        combined = torch.cat([trans_feat, gnn_feat, sent_feat], dim=-1)
        
        # Dynamic fusion layer creation
        combined_dim = combined.size(-1)
        if self.fusion is None or self._fusion_input_dim != combined_dim:
            self._fusion_input_dim = combined_dim
            
            # Build deep fusion network
            fusion_layers = []
            current_dim = combined_dim
            
            for i in range(self._num_fusion_layers):
                target_dim = self._hidden_dim if i == 0 else self._hidden_dim // (2 ** i)
                target_dim = max(target_dim, self._hidden_dim // 2)  # Don't go below hidden_dim//2
                
                fusion_layers.extend([
                    nn.Linear(current_dim, target_dim),
                    nn.LayerNorm(target_dim),
                    nn.GELU(),
                    nn.Dropout(self._dropout)
                ])
                current_dim = target_dim
            
            self.fusion = nn.Sequential(*fusion_layers).to(combined.device)
            
            for module in self.fusion.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        fused = self.fusion(combined)

        next_price = self.price_head(fused)
        multistep = self.multistep_head(fused)
        movements = self.movement_head(fused)
        directions = self.direction_head(fused).view(-1, 10, 2)
        
        range_raw = self.range_head(fused).view(-1, 10, 2)
        center = range_raw[..., 0]
        halfwidth = F.softplus(range_raw[..., 1])
        lower = center - halfwidth
        upper = center + halfwidth
        ranges = torch.stack([lower, upper], dim=-1)
        
        horizons = self.horizon_head(fused)

        return {
            'next_price': next_price,
            'multistep': multistep,
            'movements': movements,
            'directions': directions,
            'ranges': ranges,
            'horizons': horizons
        }


# ============================================================================
# DATA COLLECTION & PREPROCESSING
# ============================================================================

class NewsAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    def get_nasdaq_news(self, ticker):
        """Get news from NASDAQ"""
        articles = []
        try:
            # Clean ticker for NASDAQ (remove special characters)
            clean_ticker = ticker.replace('-USD', '').replace('=X', '').replace('^', '')
            url = f"https://www.nasdaq.com/market-activity/stocks/{clean_ticker}/news-headlines"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            for item in soup.find_all(['a', 'h3', 'div'], class_=['jupiter22-c-article-list__headline', 'news-headline']):
                text = item.get_text().strip()
                if text and 15 < len(text) < 250:
                    if text not in articles:
                        articles.append(text)
                        if len(articles) >= 10:
                            break
        except:
            pass
        return articles
    
    def get_finviz_news(self, ticker):
        """Get news from Finviz"""
        articles = []
        try:
            clean_ticker = ticker.replace('-USD', '').replace('=X', '').replace('^', '')
            url = f"https://finviz.com/quote.ashx?t={clean_ticker}"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_table = soup.find('table', {'id': 'news-table'})
            if news_table:
                for row in news_table.find_all('tr')[:10]:
                    link = row.find('a')
                    if link:
                        headline = link.get_text().strip()
                        if headline and len(headline) > 10:
                            articles.append(headline)
        except:
            pass
        return articles
    
    def get_yahoo_news(self, ticker):
        """Get news from Yahoo Finance"""
        articles = []
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if 'news' in link.get('href', '') or 'story' in link.get('href', ''):
                    text = link.get_text().strip()
                    if text and 15 < len(text) < 200:
                        if text not in articles:
                            articles.append(text)
                            if len(articles) >= 10:
                                break
        except:
            pass
        return articles
    
    def calculate_price_sentiment(self, prices, volumes):
        """Calculate sentiment from price action"""
        try:
            if len(prices) < 5:
                return 0.0
            
            # Handle NaN values
            prices = prices[~np.isnan(prices)]
            volumes = volumes[~np.isnan(volumes)]
            
            if len(prices) < 5 or len(volumes) < 5:
                return 0.0
            
            recent_return = (prices[-1] - prices[-5]) / (prices[-5] + 1e-8)
            volatility = np.std(prices[-20:]) / (np.mean(prices[-20:]) + 1e-8)
            vol_ratio = volumes[-1] / (np.mean(volumes[-5:]) + 1e-8)
            
            sentiment = np.tanh(recent_return * 10)
            sentiment *= (1 + 0.2 * (vol_ratio - 1))
            
            result = np.clip(sentiment, -1, 1)
            
            # Ensure not NaN
            if np.isnan(result):
                return 0.0
            
            return float(result)
        except:
            return 0.0
    
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def get_stock_sentiment(self, ticker, prices=None, volumes=None):
        """Get sentiment from multiple sources with fallback"""
        articles = []
        
        # Try multiple sources in order
        sources = [
            ('NASDAQ', self.get_nasdaq_news),
            ('Finviz', self.get_finviz_news),
            ('Yahoo', self.get_yahoo_news)
        ]
        
        for source_name, source_func in sources:
            if not articles:
                articles = source_func(ticker)
                if articles:
                    sentiments = [self.analyze_sentiment(a) for a in articles]
                    avg_sentiment = np.mean(sentiments)
                    print(f"    {source_name}: {len(articles)} articles, sentiment: {avg_sentiment:.3f}")
                    return avg_sentiment
        
        # Fallback to price-based sentiment
        if prices is not None and volumes is not None:
            price_sentiment = self.calculate_price_sentiment(prices, volumes)
            print(f"    No news found, price-based sentiment: {price_sentiment:.3f}")
            return price_sentiment
        
        random_sentiment = np.random.uniform(-0.15, 0.15)
        print(f"    Fallback random sentiment: {random_sentiment:.3f}")
        return random_sentiment


def download_stock_data(tickers, period='max'):
    """Download stock data for multiple tickers"""
    print(f"\nDownloading maximum available data for {len(tickers)} stocks...")
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if len(hist) > 50:
                data[ticker] = hist
                print(f"✓ {ticker}: {len(hist)} days (from {hist.index[0].date()} to {hist.index[-1].date()})")
        except Exception as e:
            print(f"✗ {ticker}: Failed - {str(e)[:50]}")
    return data


def create_transformer_features(df, seq_len=256):
    """Create features for transformer model"""
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['High_Low'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open'] = (df['Close'] - df['Open']) / df['Open']
    df = df.dropna()
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'High_Low', 'Close_Open']
    return df[feature_cols].values, df


def create_gnn_features(prices, volumes):
    """Create features for GNN model"""
    features = []
    for i in range(len(prices)):
        stock_features = [
            prices[i][-1],
            np.mean(prices[i][-5:]),
            np.mean(prices[i][-10:]),
            np.std(prices[i][-20:]),
            volumes[i][-1] / (np.mean(volumes[i][-20:]) + 1e-8)
        ]
        features.append(stock_features)
    return np.array(features)


def create_correlation_graph(prices, threshold=0.5):
    """Create graph edges based on correlation"""
    prices = np.array(prices)
    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D array (stocks × time), got shape {prices.shape}")
    if prices.shape[0] < 2:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return edge_index, torch.tensor([1.0])
    
    corr_matrix = np.corrcoef(prices)
    edge_index = []
    edge_weight = []
    
    for i in range(len(prices)):
        for j in range(len(prices)):
            if i != j and abs(corr_matrix[i, j]) > threshold:
                edge_index.append([i, j])
                edge_weight.append(abs(corr_matrix[i, j]))
    
    if not edge_index:
        edge_index = [[i, i] for i in range(len(prices))]
        edge_weight = [1.0] * len(prices)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index, torch.tensor(edge_weight, dtype=torch.float)


def create_sentiment_features(prices, volumes, sentiments, seq_len=20):
    """Create features for sentiment model"""
    features = []
    for i in range(len(prices)):
        price_window = prices[i][-seq_len:]
        volume_window = volumes[i][-seq_len:]
        
        seq_features = []
        for j in range(len(price_window)):
            feat = [
                price_window[j],
                np.mean(price_window[max(0, j-5):j+1]),
                np.std(price_window[max(0, j-5):j+1]) if j > 0 else 0,
                volume_window[j],
                sentiments[i]
            ]
            seq_features.append(feat)
        features.append(seq_features)
    
    return np.array(features)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_consistency_loss(outputs, weights=None):
    """Compute consistency loss between prediction heads"""
    if weights is None:
        weights = {'consistency': 0.3}
    
    consistency_loss = 0.0
    
    price_delta_1 = outputs['movements'][:, 0]
    price_delta_multistep = outputs['multistep'][:, 0]
    
    direction_logits = outputs['directions'][:, 0, :]
    direction_probs = F.softmax(direction_logits, dim=-1)
    
    delta_prob_up = torch.sigmoid(price_delta_1 * 10)
    delta_prob_up_multistep = torch.sigmoid(price_delta_multistep * 10)
    
    target_dist_1 = torch.stack([1 - delta_prob_up, delta_prob_up], dim=-1)
    target_dist_2 = torch.stack([1 - delta_prob_up_multistep, delta_prob_up_multistep], dim=-1)
    
    eps = 1e-8
    target_dist_1 = torch.clamp(target_dist_1, eps, 1 - eps)
    target_dist_2 = torch.clamp(target_dist_2, eps, 1 - eps)
    direction_probs = torch.clamp(direction_probs, eps, 1 - eps)
    
    kl_loss_1 = F.kl_div(torch.log(direction_probs + eps), target_dist_1, reduction='batchmean')
    kl_loss_2 = F.kl_div(torch.log(direction_probs + eps), target_dist_2, reduction='batchmean')
    
    consistency_loss += 0.5 * (kl_loss_1 + kl_loss_2)
    
    multistep_signs = torch.sign(outputs['multistep'])
    movement_signs = torch.sign(outputs['movements'])
    
    num_days = min(10, outputs['multistep'].shape[1])
    for day in range(num_days):
        day_direction_logits = outputs['directions'][:, day, :]
        day_direction_probs = F.softmax(day_direction_logits, dim=-1)
        day_direction_pred = torch.argmax(day_direction_probs, dim=-1)
        day_direction_sign = 2 * day_direction_pred - 1
        
        day_movement_sign = movement_signs[:, day]
        day_multistep_sign = multistep_signs[:, day]
        
        sign_match_movement = day_movement_sign * day_direction_sign
        sign_match_multistep = day_multistep_sign * day_direction_sign
        
        hinge_loss_movement = F.relu(1 - sign_match_movement).mean()
        hinge_loss_multistep = F.relu(1 - sign_match_multistep).mean()
        
        consistency_loss += 0.1 * (hinge_loss_movement + hinge_loss_multistep) / num_days
    
    return consistency_loss * weights.get('consistency', 0.3)


def create_training_sequences(features, targets, seq_len=256):
    """Create sequences for training"""
    X, y = [], []
    for i in range(len(features) - seq_len - 10):
        X.append(features[i:i+seq_len])
        next_prices = targets[i+seq_len:i+seq_len+10]
        current_price = targets[i+seq_len-1]
        returns = (next_prices - current_price) / current_price
        y.append(returns)
    
    return np.array(X), np.array(y)


# ============================================================================
# PREDICTION & VISUALIZATION
# ============================================================================

def format_predictions(predictions, current_price, ticker, actual_prices=None, actual_highs=None, actual_lows=None):
    """Format all 5 types of predictions"""
    
    print("\n" + "="*80)
    print(f"📊 MULTIMODAL PREDICTION RESULTS FOR {ticker}")
    print(f"Current Price: ${current_price:.2f}")
    print("="*80)
    
    results = {}
    
    # 1. Predicted closing prices
    print("\n" + "─"*80)
    print("1️⃣  PREDICTED CLOSING PRICES FOR NEXT N DAYS")
    print("─"*80)
    multistep_returns = predictions['multistep'][0].detach().cpu().numpy()
    multistep_prices = current_price * (1 + multistep_returns)
    
    num_predictions = len(multistep_prices)
    idx_0 = 0 if num_predictions > 0 else 0
    idx_4 = min(4, num_predictions - 1) if num_predictions > 0 else 0
    idx_9 = min(9, num_predictions - 1) if num_predictions > 0 else 0
    
    print(f"\nTomorrow:      ${multistep_prices[idx_0]:.2f}")
    print(f"In 5 days:     ${multistep_prices[idx_4]:.2f}")
    print(f"In 10 days:    ${multistep_prices[idx_9]:.2f}")
    
    if actual_prices is not None:
        print(f"\nACTUAL VALUES (for comparison):")
        actual_idx_0 = min(0, len(actual_prices) - 1)
        actual_idx_4 = min(4, len(actual_prices) - 1)
        actual_idx_9 = min(9, len(actual_prices) - 1)
        print(f"Tomorrow:      ${actual_prices[actual_idx_0]:.2f}  (Error: ${abs(multistep_prices[idx_0]-actual_prices[actual_idx_0]):.2f})")
        print(f"In 5 days:     ${actual_prices[actual_idx_4]:.2f}  (Error: ${abs(multistep_prices[idx_4]-actual_prices[actual_idx_4]):.2f})")
        print(f"In 10 days:    ${actual_prices[actual_idx_9]:.2f}  (Error: ${abs(multistep_prices[idx_9]-actual_prices[actual_idx_9]):.2f})")
    
    print("\n💡 Use for: swing trading, medium-term planning, backtesting profit targets")
    results['prices'] = multistep_prices
    
    # 2. Price movements
    print("\n" + "─"*80)
    print("2️⃣  PREDICTED PRICE MOVEMENTS (PERCENTAGE CHANGES)")
    print("─"*80)
    movements = predictions['movements'][0].detach().cpu().numpy() * 100
    
    print(f"\nTomorrow:      {movements[0]:+.2f}%")
    print(f"In 3 days:     {movements[2]:+.2f}%")
    print(f"In 10 days:    {movements[9]:+.2f}%")
    
    if actual_prices is not None:
        actual_movements = [(actual_prices[i] - current_price) / current_price * 100 for i in [0, 2, 9]]
        print(f"\nACTUAL MOVEMENTS:")
        print(f"Tomorrow:      {actual_movements[0]:+.2f}%")
        print(f"In 3 days:     {actual_movements[1]:+.2f}%")
        print(f"In 10 days:    {actual_movements[2]:+.2f}%")
    
    print("\n💡 Use for: risk models, position sizing, volatility-adjusted strategies")
    results['movements'] = movements
    
    # 3. Directional predictions
    print("\n" + "─"*80)
    print("3️⃣  DIRECTIONAL PREDICTIONS (UP/DOWN WITH CONFIDENCE)")
    print("─"*80)
    directions = F.softmax(predictions['directions'][0], dim=-1).detach().cpu().numpy()
    
    tom_dir = "Up ⬆️" if directions[0][1] > 0.5 else "Down ⬇️"
    tom_conf = max(directions[0]) * 100
    print(f"\nTomorrow:      {tom_dir} ({tom_conf:.1f}% confidence)")
    
    week_dir = "Up ⬆️" if directions[6][1] > 0.5 else "Down ⬇️"
    week_conf = max(directions[6]) * 100
    print(f"Next week:     {week_dir} ({week_conf:.1f}% confidence)")
    
    day10_dir = "Up ⬆️" if directions[9][1] > 0.5 else "Down ⬇️"
    day10_conf = max(directions[9]) * 100
    print(f"In 10 days:    {day10_dir} ({day10_conf:.1f}% confidence)")
    
    if actual_prices is not None:
        actual_dirs = [
            "Up ⬆️" if actual_prices[0] > current_price else "Down ⬇️",
            "Up ⬆️" if actual_prices[6] > current_price else "Down ⬇️",
            "Up ⬆️" if actual_prices[9] > current_price else "Down ⬇️"
        ]
        correct = [
            tom_dir == actual_dirs[0],
            week_dir == actual_dirs[1],
            day10_dir == actual_dirs[2]
        ]
        print(f"\nACTUAL DIRECTIONS:")
        print(f"Tomorrow:      {actual_dirs[0]} {'✓' if correct[0] else '✗'}")
        print(f"Next week:     {actual_dirs[1]} {'✓' if correct[1] else '✗'}")
        print(f"In 10 days:    {actual_dirs[2]} {'✓' if correct[2] else '✗'}")
    
    print("\n💡 Use for: basic signals, ensembling with indicators, RL trading systems")
    results['directions'] = directions
    
    # 4. Price ranges
    print("\n" + "─"*80)
    print("4️⃣  PRICE RANGES (PREDICTED HIGH/LOW FOR NEXT PERIOD)")
    print("─"*80)
    ranges = predictions['ranges'][0].detach().cpu().numpy()
    
    tom_high = current_price * (1 + ranges[0][1])
    tom_low = current_price * (1 + ranges[0][0])
    
    print(f"\nTomorrow:")
    print(f"  Predicted High: ${tom_high:.2f}")
    print(f"  Predicted Low:  ${tom_low:.2f}")
    print(f"  Predicted Range: ${tom_high - tom_low:.2f} ({(tom_high-tom_low)/current_price*100:.2f}%)")
    
    if actual_prices is not None and len(actual_prices) >= 1:
        actual_tom = actual_prices[0]
        print(f"\n  Actual Close: ${actual_tom:.2f}")
        print(f"  Close Within Range: {'✓' if tom_low <= actual_tom <= tom_high else '✗'}")
        
        # Show actual high/low if available from OHLC data
        if actual_highs is not None and actual_lows is not None and len(actual_highs) >= 1:
            actual_high = actual_highs[0]
            actual_low = actual_lows[0]
            actual_range = actual_high - actual_low
            
            print(f"\n  Actual High: ${actual_high:.2f}")
            print(f"  Actual Low:  ${actual_low:.2f}")
            print(f"  Actual Range: ${actual_range:.2f} ({actual_range/current_price*100:.2f}%)")
            
            # Check if our predictions captured the actual range
            high_captured = tom_high >= actual_high
            low_captured = tom_low <= actual_low
            print(f"\n  High Prediction: {'✓ Covered' if high_captured else '✗ Underestimated'}")
            print(f"  Low Prediction:  {'✓ Covered' if low_captured else '✗ Underestimated'}")
    
    print("\n💡 Use for: risk management, stop-loss/take-profit levels, options pricing")
    results['ranges'] = ranges
    
    # 5. Multi-horizon
    print("\n" + "─"*80)
    print("5️⃣  MULTI-HORIZON PREDICTIONS (MULTIPLE TIMEFRAMES)")
    print("─"*80)
    horizons = predictions['horizons'][0].detach().cpu().numpy()
    
    day_1 = current_price * (1 + horizons[0])
    day_7 = current_price * (1 + horizons[1])
    day_30 = current_price * (1 + horizons[2]) if len(horizons) > 2 else day_7
    
    print(f"\n1-day forecast:   ${day_1:.2f}  ({(day_1-current_price)/current_price*100:+.2f}%)")
    print(f"7-day forecast:   ${day_7:.2f}  ({(day_7-current_price)/current_price*100:+.2f}%)")
    print(f"30-day forecast:  ${day_30:.2f}  ({(day_30-current_price)/current_price*100:+.2f}%)")
    
    if actual_prices is not None:
        print(f"\nACTUAL VALUES:")
        if len(actual_prices) > 0:
            print(f"1-day actual:     ${actual_prices[0]:.2f}  (Error: ${abs(day_1-actual_prices[0]):.2f})")
        if len(actual_prices) > 6:
            print(f"7-day actual:     ${actual_prices[6]:.2f}  (Error: ${abs(day_7-actual_prices[6]):.2f})")
        if len(actual_prices) > 9:
            print(f"30-day actual:    ${actual_prices[9]:.2f}  (Error: ${abs(day_30-actual_prices[9]):.2f})")
    
    print("\n💡 Use for: understanding short vs long-term trends, multi-timeframe strategies")
    results['horizons'] = [day_1, day_7, day_30]
    
    print("\n" + "="*80)
    
    return results


def visualize_predictions(results, ticker, current_price, actual_prices=None, save_dir='results'):
    """Create comprehensive visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Multimodal Predictions for {ticker}', fontsize=16, fontweight='bold')
    
    days = np.arange(1, 11)
    
    # 1. Multi-step price predictions
    ax = axes[0, 0]
    ax.plot(days, results['prices'], 'o-', linewidth=2, markersize=8, color='steelblue', label='Predicted')
    if actual_prices is not None:
        ax.plot(days, actual_prices, 's--', linewidth=2, markersize=6, color='green', label='Actual', alpha=0.7)
    ax.axhline(y=current_price, color='red', linestyle='--', label='Current Price')
    ax.set_title('10-Day Price Forecast', fontweight='bold')
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Price movements
    ax = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in results['movements']]
    ax.bar(days, results['movements'], color=colors, alpha=0.7, label='Predicted')
    if actual_prices is not None:
        actual_movements = [(actual_prices[i] - current_price) / current_price * 100 for i in range(10)]
        ax.plot(days, actual_movements, 'ko-', linewidth=2, markersize=6, label='Actual')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Predicted % Changes', fontweight='bold')
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Change (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Direction confidence
    ax = axes[0, 2]
    up_conf = results['directions'][:, 1] * 100
    down_conf = results['directions'][:, 0] * 100
    ax.plot(days, up_conf, 'o-', label='Up Confidence', color='green', linewidth=2)
    ax.plot(days, down_conf, 'o-', label='Down Confidence', color='red', linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Directional Confidence', fontweight='bold')
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Confidence (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Price ranges
    ax = axes[1, 0]
    highs = current_price * (1 + results['ranges'][:, 1])
    lows = current_price * (1 + results['ranges'][:, 0])
    ax.fill_between(days, lows, highs, alpha=0.3, color='steelblue', label='Predicted Range')
    ax.plot(days, results['prices'], 'o-', color='darkblue', linewidth=2, label='Mid Price')
    if actual_prices is not None:
        ax.plot(days, actual_prices, 's-', color='green', linewidth=2, label='Actual', alpha=0.7)
    ax.axhline(y=current_price, color='red', linestyle='--', label='Current')
    ax.set_title('Predicted Price Ranges', fontweight='bold')
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Multi-horizon comparison
    ax = axes[1, 1]
    horizons = ['1 Day', '1 Week', '1 Month']
    horizon_prices = results['horizons']
    changes = [(p - current_price) / current_price * 100 for p in horizon_prices]
    colors_h = ['green' if c > 0 else 'red' for c in changes]
    bars = ax.bar(horizons, changes, color=colors_h, alpha=0.7, label='Predicted')
    
    if actual_prices is not None and len(actual_prices) >= 10:
        actual_changes = [
            (actual_prices[0] - current_price) / current_price * 100,
            (actual_prices[6] - current_price) / current_price * 100,
            (actual_prices[9] - current_price) / current_price * 100
        ]
        ax.scatter([0, 1, 2], actual_changes, color='black', s=100, marker='X', label='Actual', zorder=5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Multi-Horizon % Changes', fontweight='bold')
    ax.set_ylabel('Change (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for bar, change in zip(bars, changes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{change:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    📊 PREDICTION SUMMARY
    
    Current Price: ${current_price:.2f}
    
    Next Day:
      Price: ${results['prices'][0]:.2f}
      Change: {results['movements'][0]:+.2f}%
    
    Next Week:
      Price: ${results['horizons'][1]:.2f}
      Change: {(results['horizons'][1]-current_price)/current_price*100:+.2f}%
    
    Next Month:
      Price: ${results['horizons'][2]:.2f}
      Change: {(results['horizons'][2]-current_price)/current_price*100:+.2f}%
    
    Risk Range (Tomorrow):
      High: ${highs[0]:.2f}
      Low:  ${lows[0]:.2f}
    """
    
    if actual_prices is not None:
        mape = np.mean(np.abs((actual_prices - results['prices']) / actual_prices)) * 100
        rmse = np.sqrt(np.mean((actual_prices - results['prices'])**2))
        summary_text += f"""
    
    📈 ACCURACY METRICS:
      MAPE: {mape:.2f}%
      RMSE: ${rmse:.2f}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    filename = f'{save_dir}/{ticker}_multimodal_predictions.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to '{filename}'")
    plt.show()
    
    return filename


def save_predictions_to_csv(results, ticker, current_price, actual_prices=None, save_dir='results'):
    """Save predictions to CSV"""
    os.makedirs(save_dir, exist_ok=True)
    
    data = {
        'Day': list(range(1, 11)),
        'Predicted_Price': results['prices'],
        'Predicted_Change_%': results['movements'],
        'Up_Confidence_%': results['directions'][:, 1] * 100,
        'Down_Confidence_%': results['directions'][:, 0] * 100,
        'Predicted_High': current_price * (1 + results['ranges'][:, 1]),
        'Predicted_Low': current_price * (1 + results['ranges'][:, 0])
    }
    
    if actual_prices is not None:
        data['Actual_Price'] = actual_prices
        data['Price_Error_USD'] = results['prices'] - actual_prices
        data['Price_Error_Percent'] = (results['prices'] - actual_prices) / actual_prices * 100
    
    df = pd.DataFrame(data)
    filename = f'{save_dir}/{ticker}_predictions.csv'
    df.to_csv(filename, index=False)
    print(f"✓ Predictions saved to '{filename}'")
    
    return filename


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 MULTIMODAL STOCK PREDICTION SYSTEM")
    print("Combining: Transformer + GNN + Sentiment Analysis")
    print("="*80)
    
    print("\n" + "="*80)
    print("📝 OPTIMIZATION GUIDE")
    print("="*80)
    print("""
To make the model larger and more powerful, edit these parameters in the code:

1. TRANSFORMER (line ~95):
   - d_model: 128 → 256, 512 (hidden size)
   - nhead: 8 → 16, 32 (attention heads)
   - num_layers: 4 → 6, 8, 12 (depth)
   - seq_len: 256 → 512, 1024 (sequence length)

2. GNN (line ~137):
   - hidden_dim: 128 → 256, 512 (node embeddings)
   - num_gat_layers: 4 → 6, 8 (graph depth)
   - heads: 8 → 16 (attention heads per layer)

3. SENTIMENT LSTM (line ~181):
   - hidden_dim: 128 → 256, 512 (LSTM hidden size)
   - num_layers: 3 → 4, 6 (LSTM depth)
   - num_heads: 8 → 16 (attention heads)

4. FUSION NETWORK (line ~219):
   - hidden_dim: 256 → 512, 1024 (fusion layer size)
   - num_fusion_layers: 3 → 4, 6 (fusion depth)

5. TRAINING (in main):
   - epochs: 50 → 100, 200 (training iterations)
   - lr: 0.001 → 0.0001, 0.0005 (learning rate)
   - batch_size: 16 → 32, 64 (samples per batch)

TIPS:
- Start with 2x increases, monitor GPU memory
- Larger models need more data and training time
- Reduce learning rate when increasing model size
- Use gradient accumulation for larger effective batch sizes
    """)
    print("="*80)
    
    # USER INPUT 1: Stock ticker
    ticker = input("\nEnter stock ticker symbol (e.g., AAPL, TSLA, BTC-USD, EURUSD=X): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
        print(f"No ticker entered. Using default: {ticker}")
    
    # USER INPUT 2: Load existing model
    load_existing = input("\nDo you want to load an existing model? (yes/no): ").strip().lower()
    load_model = load_existing in ['yes', 'y', '1', 'true']
    
    # USER INPUT 3: Backtest or predict future
    mode = input("\nChoose mode:\n  1. Backtest on historical data (test accuracy)\n  2. Predict future prices (live prediction)\nEnter choice (1 or 2): ").strip()
    backtest_mode = mode == '1'
    
    if backtest_mode:
        print("\n📊 BACKTEST MODE: Will test predictions against historical data")
    else:
        print("\n🔮 PREDICTION MODE: Will predict future stock movements")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define related stocks for GNN
    sector_stocks = {
        'AAPL': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        'TSLA': ['TSLA', 'F', 'GM', 'RIVN', 'NIO', 'LCID', 'AAPL'],
        'GOOGL': ['GOOGL', 'META', 'AMZN', 'MSFT', 'AAPL', 'NFLX', 'NVDA'],
        'MSFT': ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ORCL'],
        'NVDA': ['NVDA', 'AMD', 'INTC', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'BTC-USD': ['BTC-USD', 'ETH-USD', 'COIN', 'MSTR', 'SQ', 'PYPL'],
        'ETH-USD': ['ETH-USD', 'BTC-USD', 'COIN', 'MSTR', 'SQ', 'PYPL'],
        'EURUSD=X': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'DX-Y.NYB'],
        'GBPUSD=X': ['GBPUSD=X', 'EURUSD=X', 'USDJPY=X', 'DX-Y.NYB'],
    }
    
    related_tickers = sector_stocks.get(ticker, [ticker, 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'])
    if ticker not in related_tickers:
        related_tickers[0] = ticker
    
    print(f"\nTarget Stock: {ticker}")
    print(f"Related Stocks (for GNN): {', '.join(related_tickers)}")
    
    # Download maximum available data
    stock_data = download_stock_data(related_tickers, period='max')
    
    if len(stock_data) < 1:
        print("\nError: Could not download data for any stocks")
        exit()
    
    if ticker not in stock_data:
        print(f"\nError: Could not download data for {ticker}")
        exit()
    
    # Get sentiment data
    print("\nFetching sentiment data (Yahoo PRIMARY, then NASDAQ)...")
    analyzer = NewsAnalyzer()
    sentiments = {}
    for t in stock_data.keys():
        print(f"\n  {t}:")
        prices = stock_data[t]['Close'].values
        volumes = stock_data[t]['Volume'].values
        sent = analyzer.get_stock_sentiment(t, prices=prices, volumes=volumes)
        sentiments[t] = sent
    
    # Prepare data
    print("\n" + "="*80)
    print("PREPARING DATA...")
    print("="*80)
    
    # Get common dates - with special handling for crypto (24/7) vs stocks (weekdays)
    all_dates = None
    date_sets = {}
    
    for t, df in stock_data.items():
        date_sets[t] = set(df.index.date)  # Use date only, ignore time
        if all_dates is None:
            all_dates = date_sets[t]
        else:
            all_dates = all_dates.intersection(date_sets[t])
    
    # If no common dates (crypto + stocks), use a more flexible approach
    if len(all_dates) == 0:
        print("\n⚠️  No exact date overlap (likely mixing crypto 24/7 with stock market hours)")
        print("  Using flexible date alignment (matching closest weekdays)...")
        
        # Get all unique dates across all tickers
        all_unique_dates = set()
        for dates in date_sets.values():
            all_unique_dates.update(dates)
        
        # Sort and use dates that appear in at least 50% of tickers
        from collections import Counter
        date_counts = Counter()
        for dates in date_sets.values():
            for date in dates:
                date_counts[date] += 1
        
        # Use dates that appear in at least half the tickers
        threshold = len(stock_data) // 2
        common_dates = sorted([date for date, count in date_counts.items() if count >= threshold])
        
        if len(common_dates) == 0:
            print("\n❌ Error: Still no overlapping dates after flexible matching")
            print("\nDebugging info:")
            for t, dates in date_sets.items():
                print(f"  {t}: {len(dates)} days, range {min(dates)} to {max(dates)}")
            print("\nTip: Use either all crypto OR all stocks, don't mix them")
            exit()
    else:
        common_dates = sorted(list(all_dates))
    
    # Convert back to timestamps for indexing
    common_dates_timestamps = []
    for t in stock_data.keys():
        df = stock_data[t]
        matching_dates = [ts for ts in df.index if ts.date() in set(common_dates)]
        if not common_dates_timestamps:
            common_dates_timestamps = matching_dates
        else:
            # Keep only dates that exist in this ticker too
            common_dates_timestamps = [ts for ts in common_dates_timestamps if ts.date() in [d.date() for d in matching_dates]]
    
    common_dates = sorted(common_dates_timestamps)
    
    if len(common_dates) == 0:
        print("\n❌ Error: No common trading dates found between stocks")
        print("This usually happens when:")
        print("  1. Crypto (24/7) mixed with stocks (weekdays only)")
        print("  2. Stocks are from different exchanges")
        print("  3. Some tickers have insufficient data")
        print("\nTip: Use stocks from the same exchange (e.g., all US stocks or all crypto)")
        exit()
    
    print(f"Total trading days: {len(common_dates)}")
    print(f"Date range: {common_dates[0].date()} to {common_dates[-1].date()}")
    
    # Extract data for all stocks - FIX timezone issues
    prices_dict = {}
    volumes_dict = {}
    for t in stock_data.keys():
        df = stock_data[t]
        # Remove timezone info for compatibility
        df.index = df.index.tz_localize(None)
        
        # Filter to common dates (also remove timezone from common_dates)
        common_dates_no_tz = pd.DatetimeIndex([d.tz_localize(None) if hasattr(d, 'tz_localize') else d for d in common_dates])
        
        # Use intersection to get matching dates
        matching_dates = df.index.intersection(common_dates_no_tz)
        df_filtered = df.loc[matching_dates]
        
        prices_dict[t] = df_filtered['Close'].values
        volumes_dict[t] = df_filtered['Volume'].values
    
    # Update common_dates to the actual filtered dates
    # Use the first ticker's matching dates as reference
    first_ticker = list(stock_data.keys())[0]
    common_dates = stock_data[first_ticker].index.intersection(common_dates_no_tz).tolist()
    
    # Get target stock data
    target_df = stock_data[ticker].loc[common_dates]
    target_idx = list(stock_data.keys()).index(ticker)
    
    # Determine split point for backtest mode
    if backtest_mode:
        split_idx = int(0.8 * len(common_dates))
        test_dates = common_dates[split_idx:]
        train_dates = common_dates[:split_idx]
        
        current_price = float(target_df['Close'].iloc[split_idx-1])
        
        # Get actual future prices AND high/low data for backtesting
        actual_future_prices = target_df['Close'].iloc[split_idx:split_idx+10].values
        actual_future_highs = target_df['High'].iloc[split_idx:split_idx+10].values if 'High' in target_df.columns else None
        actual_future_lows = target_df['Low'].iloc[split_idx:split_idx+10].values if 'Low' in target_df.columns else None
        
        print(f"\n📊 Backtest Configuration:")
        print(f"  Training period: {train_dates[0].date()} to {train_dates[-1].date()}")
        print(f"  Test period: {test_dates[0].date()} to {test_dates[min(9, len(test_dates)-1)].date()}")
        print(f"  Test start price: ${current_price:.2f}")
        
        target_df_train = target_df.iloc[:split_idx]
    else:
        current_price = float(target_df['Close'].iloc[-1])
        actual_future_prices = None
        actual_future_highs = None
        actual_future_lows = None
        target_df_train = target_df
        print(f"\n🔮 Prediction from: {common_dates[-1].date()}")
        print(f"  Current price: ${current_price:.2f}")
    
    # Create features for each model
    print("\nCreating features for each model component...")
    
    # 1. Transformer features
    trans_features, _ = create_transformer_features(target_df_train, seq_len=256)
    trans_scaler = MinMaxScaler()
    trans_features_scaled = trans_scaler.fit_transform(trans_features)
    
    if len(trans_features_scaled) < 60:
        print(f"\nError: Not enough data for transformer (need 60+ days, have {len(trans_features_scaled)})")
        exit()
    
    trans_input = torch.FloatTensor(trans_features_scaled[-60:]).unsqueeze(0)
    print(f"  ✓ Transformer features: {trans_input.shape}")
    
    # 2. GNN features
    if backtest_mode:
        min_len = min(60, split_idx)
        prices_array = np.array([prices_dict[t][max(0, split_idx-min_len):split_idx] for t in stock_data.keys()])
        volumes_array = np.array([volumes_dict[t][max(0, split_idx-min_len):split_idx] for t in stock_data.keys()])
    else:
        min_len = min(60, len(prices_dict[list(stock_data.keys())[0]]))
        prices_array = np.array([prices_dict[t][-min_len:] for t in stock_data.keys()])
        volumes_array = np.array([volumes_dict[t][-min_len:] for t in stock_data.keys()])
    
    gnn_features = create_gnn_features(prices_array, volumes_array)
    gnn_scaler = StandardScaler()
    gnn_features_scaled = gnn_scaler.fit_transform(gnn_features)
    gnn_input = torch.FloatTensor(gnn_features_scaled)
    
    if backtest_mode:
        graph_prices = np.array([prices_dict[t][:split_idx] for t in stock_data.keys()])
    else:
        graph_prices = prices_array
    edge_index, _ = create_correlation_graph(graph_prices, threshold=0.5)
    print(f"  ✓ GNN features: {gnn_input.shape}, Edges: {edge_index.shape[1]}")
    
    # 3. Sentiment features
    sentiment_array = np.array([sentiments[t] for t in stock_data.keys()])
    sent_features = create_sentiment_features(prices_array, volumes_array, sentiment_array, seq_len=20)
    sent_scaler = StandardScaler()
    sent_features_flat = sent_features[target_idx].reshape(-1, sent_features.shape[-1])
    sent_features_scaled = sent_scaler.fit_transform(sent_features_flat)
    sent_input = torch.FloatTensor(sent_features_scaled).unsqueeze(0)
    print(f"  ✓ Sentiment features: {sent_input.shape}")
    
    target_gnn_features = gnn_input[target_idx:target_idx+1]
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = MultimodalStockPredictor(
        transformer_features=trans_input.shape[-1],
        gnn_features=gnn_input.shape[-1],
        sentiment_features=sent_input.shape[-1],
        hidden_dim=256,
        num_fusion_layers=3,
        seq_len=256
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    
    # Test that all three models work
    print("\n" + "="*80)
    print("🧪 TESTING MODEL COMPONENTS")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        try:
            # Test transformer
            trans_out = model.transformer(trans_input.to(device))
            print(f"✓ Transformer working: output shape {trans_out.shape}")
            
            # Test GNN
            gnn_out = model.gnn(gnn_input.to(device), edge_index.to(device))
            print(f"✓ GNN working: output shape {gnn_out.shape}")
            
            # Test sentiment
            sent_out = model.sentiment(sent_input.to(device))
            print(f"✓ Sentiment LSTM working: output shape {sent_out.shape}")
            
            # Test full forward pass
            test_pred = model(
                trans_input.to(device),
                gnn_input.to(device),
                edge_index.to(device),
                sent_input.to(device),
                target_idx
            )
            print(f"✓ Full multimodal forward pass working")
            print(f"  - Next price: {test_pred['next_price'].shape}")
            print(f"  - Multistep: {test_pred['multistep'].shape}")
            print(f"  - Movements: {test_pred['movements'].shape}")
            print(f"  - Directions: {test_pred['directions'].shape}")
            print(f"  - Ranges: {test_pred['ranges'].shape}")
            print(f"  - Horizons: {test_pred['horizons'].shape}")
            print("✅ All components verified and working correctly!")
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            import traceback
            traceback.print_exc()
            exit()
    
    # Load or train model
    model_path = f'models/multimodal_{ticker}.pth'
    
    if load_model and os.path.exists(model_path):
        print(f"\n✓ Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        if load_model:
            print(f"\n⚠️  No saved model found at {model_path}")
        
        print("\n" + "="*80)
        print("TRAINING NEW MODEL")
        print("="*80)
        print("Note: Training will continue until validation loss stops improving\n")
        
        # Prepare training data
        train_features, _ = create_transformer_features(target_df_train, seq_len=256)
        train_features_scaled = trans_scaler.transform(train_features)
        
        X_seq, y_seq = create_training_sequences(
            train_features_scaled, 
            target_df_train['Close'].values, 
            seq_len=256
        )
        
        if len(X_seq) < 20:
            print("\n⚠️  Not enough data for proper training. Using model with random initialization.")
            print("   Recommendation: Use a stock with more historical data.")
        else:
            split = int(0.8 * len(X_seq))
            X_train_seq = torch.FloatTensor(X_seq[:split])
            y_train_seq = torch.FloatTensor(y_seq[:split])
            X_val_seq = torch.FloatTensor(X_seq[split:])
            y_val_seq = torch.FloatTensor(y_seq[split:])
            
            class MultimodalDataset(torch.utils.data.Dataset):
                def __init__(self, X, y, gnn_input, sent_input, edge_index, target_idx):
                    self.X = X
                    self.y = y
                    self.gnn_input = gnn_input
                    self.sent_input = sent_input
                    self.edge_index = edge_index
                    self.target_idx = target_idx
                
                def __len__(self):
                    return len(self.X)
                
                def __getitem__(self, idx):
                    return {
                        'trans_x': self.X[idx],
                        'y': self.y[idx]
                    }
            
            train_dataset = MultimodalDataset(X_train_seq, y_train_seq, gnn_input, sent_input, edge_index, target_idx)
            val_dataset = MultimodalDataset(X_val_seq, y_val_seq, gnn_input, sent_input, edge_index, target_idx)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)
            
            print(f"Training samples: {len(X_train_seq)}")
            print(f"Validation samples: {len(X_val_seq)}")
            print("\nStarting training...")
            
            loss_weights = {
                "price": 1.0,
                "multistep": 1.0,
                "movement": 1.0,
                "direction": 1.0,
                "range": 1.0,
                "horizon": 1.0,
                "consistency": 0.3
            }
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            epochs = 50
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                batch_count = 0
                
                # Training progress bar
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                                 leave=False, ncols=100, file=sys.stdout, dynamic_ncols=True)
                
                for batch in train_pbar:
                    
                    batch_x = batch['trans_x'].to(device)
                    batch_y = batch['y'].to(device)
                    
                    batch_size = batch_x.size(0)
                    sent_batch = sent_input.repeat(batch_size, 1, 1).to(device)
                    edge_batch = edge_index.to(device)
                    gnn_full = gnn_input.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x, gnn_full, edge_batch, sent_batch, target_idx)
                    
                    targets_next_price = batch_y[:, 0:1]
                    targets_multistep = batch_y
                    targets_movements = batch_y
                    
                    targets_directions = (batch_y > 0).long()
                    
                    returns_std = torch.std(batch_y, dim=1, keepdim=True)
                    margin = torch.clamp(returns_std * 2.0, min=0.01, max=0.1)
                    targets_ranges_lower = batch_y - margin
                    targets_ranges_upper = batch_y + margin
                    targets_ranges = torch.stack([targets_ranges_lower, targets_ranges_upper], dim=-1)
                    
                    targets_horizon = torch.stack([
                        batch_y[:, 0],
                        batch_y[:, min(6, batch_y.shape[1]-1)],
                        batch_y[:, min(9, batch_y.shape[1]-1)]
                    ], dim=-1)
                    
                    total_loss = 0.0
                    
                    loss_price = F.mse_loss(outputs['next_price'], targets_next_price)
                    total_loss += loss_weights['price'] * loss_price
                    
                    loss_multistep = F.mse_loss(outputs['multistep'], targets_multistep)
                    total_loss += loss_weights['multistep'] * loss_multistep
                    
                    loss_movement = F.mse_loss(outputs['movements'], targets_movements)
                    total_loss += loss_weights['movement'] * loss_movement
                    
                    directions_flat = outputs['directions'].view(-1, 2)
                    targets_directions_flat = targets_directions.view(-1)
                    loss_direction = F.cross_entropy(directions_flat, targets_directions_flat)
                    total_loss += loss_weights['direction'] * loss_direction
                    
                    loss_range = F.smooth_l1_loss(outputs['ranges'], targets_ranges)
                    total_loss += loss_weights['range'] * loss_range
                    
                    loss_horizon = F.mse_loss(outputs['horizons'], targets_horizon)
                    total_loss += loss_weights['horizon'] * loss_horizon
                    
                    loss_consistency = compute_consistency_loss(outputs, loss_weights)
                    total_loss += loss_consistency
                    
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                    batch_count += 1
                    
                    # Update progress bar with current loss
                    train_pbar.set_postfix({'loss': f'{total_loss.item():.4f}'}, refresh=True)
                
                train_pbar.close()
                avg_train_loss = train_loss / len(train_loader)
                
                model.eval()
                val_loss = 0
                
                # Validation progress bar
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', 
                               leave=False, ncols=100, file=sys.stdout, dynamic_ncols=True)
                
                with torch.no_grad():
                    for batch in val_pbar:
                        batch_x = batch['trans_x'].to(device)
                        batch_y = batch['y'].to(device)
                        batch_size = batch_x.size(0)
                        
                        sent_batch = sent_input.repeat(batch_size, 1, 1).to(device)
                        edge_batch = edge_index.to(device)
                        gnn_full = gnn_input.to(device)
                        
                        outputs = model(batch_x, gnn_full, edge_batch, sent_batch, target_idx)
                        
                        targets_multistep = batch_y
                        loss = F.mse_loss(outputs['multistep'], targets_multistep)
                        val_loss += loss.item()
                        
                        # Update progress bar with current loss
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=True)
                
                val_pbar.close()
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                # Print epoch summary
                status = ""
                if avg_val_loss < best_val_loss:
                    status = "✓ BEST"
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} {status}")
                
                if patience_counter >= max_patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"\n✓ Training complete! Best validation loss: {best_val_loss:.6f}")
    
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
    
    scalers = {
        'trans_scaler': trans_scaler,
        'gnn_scaler': gnn_scaler,
        'sent_scaler': sent_scaler,
        'ticker': ticker,
        'related_tickers': list(stock_data.keys()),
        'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f'models/scalers_{ticker}.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print(f"✓ Model and scalers saved")
    
    # Make predictions
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS...")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        predictions = model(
            trans_input.to(device),
            gnn_input.to(device),
            edge_index.to(device),
            sent_input.to(device),
            target_idx
        )
    
    results = format_predictions(predictions, current_price, ticker, actual_future_prices, actual_future_highs, actual_future_lows)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'results/{ticker}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_predictions(results, ticker, current_price, actual_future_prices, save_dir)
    save_predictions_to_csv(results, ticker, current_price, actual_future_prices, save_dir)
    
    # Save summary report
    with open(f'{save_dir}/summary_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MULTIMODAL STOCK PREDICTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Target Stock: {ticker}\n")
        f.write(f"Current Price: ${current_price:.2f}\n")
        f.write(f"Mode: {'BACKTEST' if backtest_mode else 'LIVE PREDICTION'}\n")
        f.write(f"Related Stocks: {', '.join(stock_data.keys())}\n")
        f.write(f"Data Range: {common_dates[0].date()} to {common_dates[-1].date()}\n")
        f.write(f"Total Trading Days: {len(common_dates)}\n\n")
        
        f.write(f"Model Size: {total_params:,} parameters\n\n")
        
        f.write("SENTIMENT SCORES:\n")
        for t, s in sentiments.items():
            f.write(f"  {t}: {s:.3f}\n")
        f.write("\n")
        
        f.write("PREDICTIONS:\n")
        f.write(f"  Tomorrow: ${results['prices'][0]:.2f} ({results['movements'][0]:+.2f}%)\n")
        f.write(f"  5 Days: ${results['prices'][4]:.2f} ({results['movements'][4]:+.2f}%)\n")
        f.write(f"  10 Days: ${results['prices'][9]:.2f} ({results['movements'][9]:+.2f}%)\n\n")
        
        f.write("MULTI-HORIZON:\n")
        f.write(f"  1-Day: ${results['horizons'][0]:.2f}\n")
        f.write(f"  1-Week: ${results['horizons'][1]:.2f}\n")
        f.write(f"  1-Month: ${results['horizons'][2]:.2f}\n\n")
        
        if actual_future_prices is not None:
            mape = np.mean(np.abs((actual_future_prices - results['prices']) / actual_future_prices)) * 100
            rmse = np.sqrt(np.mean((actual_future_prices - results['prices'])**2))
            mae = np.mean(np.abs(actual_future_prices - results['prices']))
            
            pred_dirs = [p > current_price for p in results['prices']]
            actual_dirs = [p > current_price for p in actual_future_prices]
            dir_acc = np.mean([p == a for p, a in zip(pred_dirs, actual_dirs)]) * 100
            
            f.write("ACCURACY METRICS:\n")
            f.write(f"  MAPE: {mape:.2f}%\n")
            f.write(f"  RMSE: ${rmse:.2f}\n")
            f.write(f"  MAE: ${mae:.2f}\n")
            f.write(f"  Direction Accuracy: {dir_acc:.2f}%\n\n")
            
            f.write("ACTUAL vs PREDICTED:\n")
            for i in range(min(10, len(actual_future_prices))):
                error = results['prices'][i] - actual_future_prices[i]
                error_pct = error / actual_future_prices[i] * 100
                f.write(f"  Day {i+1}: Pred=${results['prices'][i]:.2f}, Actual=${actual_future_prices[i]:.2f}, Error=${error:+.2f} ({error_pct:+.2f}%)\n")
    
    print(f"✓ Summary report saved to '{save_dir}/summary_report.txt'")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 All results saved to: {save_dir}/")
    print(f"   • {ticker}_multimodal_predictions.png (visualizations)")
    print(f"   • {ticker}_predictions.csv (detailed predictions)")
    print(f"   • summary_report.txt (complete analysis)")
    
    if backtest_mode and actual_future_prices is not None:
        mape = np.mean(np.abs((actual_future_prices - results['prices']) / actual_future_prices)) * 100
        dir_acc = np.mean([p == a for p, a in zip(
            [p > current_price for p in results['prices']],
            [p > current_price for p in actual_future_prices]
        )]) * 100
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   • Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"   • Direction Accuracy: {dir_acc:.2f}%")
    
    print("\n" + "="*80)
    print("💡 Model Components:")
    print("   ✓ Transformer: Long-range temporal patterns")
    print("   ✓ GNN: Stock correlations and relationships")
    print("   ✓ Sentiment: News and market sentiment")
    print("   ✓ Multi-task heads: 5 prediction types")
    print("="*80)
    
    print("\n" + "="*80)
    print("🎯 OPTIMIZATION TIPS FOR BETTER PERFORMANCE:")
    print("="*80)
    print("""
1. Increase model capacity (see guide at start)
2. Train longer: epochs = 100 or 200
3. Lower learning rate: lr = 0.0001
4. Larger batch size: batch_size = 32 or 64
5. More data: Use stocks with longer history
6. Ensemble: Train multiple models and average predictions
7. Feature engineering: Add more technical indicators
8. Hyperparameter tuning: Grid search optimal parameters
    """)
    print("="*80)