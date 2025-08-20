import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from ta import add_all_ta_features
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice, AccDistIndexIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator, IchimokuIndicator
from ta.others import DailyReturnIndicator

def create_rolling_features(btc_data, window_size=96):  # 24 hours in 15-min intervals (24*4=96)
    """
    Create a comprehensive set of features using rolling windows
    """
    df = {}
    
    for key, _ in btc_data.columns:
        print(key)
        df[key] = btc_data[key].to_numpy().flatten()
    df = pd.DataFrame(df, index = btc_data.index)
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Price-based features
    for window in [4, 8, 16, 24, 48, 96]:  # 1h, 2h, 4h, 6h, 12h, 24h windows
        # Rolling statistics
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['Close'].rolling(window=window).max()
        
        # Price ratios
        df[f'price_ratio_mean_{window}'] = df['Close'] / df[f'rolling_mean_{window}']
        df[f'price_ratio_min_{window}'] = df['Close'] / df[f'rolling_min_{window}']
        df[f'price_ratio_max_{window}'] = df['Close'] / df[f'rolling_max_{window}']
        
        # Volatility features
        df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        # Return features
        df[f'cumulative_return_{window}'] = df['Close'] / df['Close'].shift(window) - 1
        
        # Volume features
        df[f'volume_mean_{window}'] = df['Volume'].rolling(window=window).mean()
        df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_mean_{window}']
    
    # Technical indicators with multiple parameter sets
    # RSI with different windows
    for window in [6, 12, 24]:
        rsi = RSIIndicator(close=df['Close'], window=window)
        df[f'rsi_{window}'] = rsi.rsi()
    
    # Bollinger Bands with different windows
    for window in [12, 24, 48]:
        bb = BollingerBands(close=df['Close'], window=window, window_dev=2)
        df[f'bb_high_{window}'] = bb.bollinger_hband()
        df[f'bb_low_{window}'] = bb.bollinger_lband()
        df[f'bb_width_{window}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        df[f'bb_position_{window}'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    # MACD with different parameter sets
    for fast in [6, 12]:
        for slow in [12, 24]:
            if fast < slow:
                macd = MACD(close=df['Close'], window_fast=fast, window_slow=slow, window_sign=9)
                df[f'macd_{fast}_{slow}'] = macd.macd()
                df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
                df[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()
    
    # Stochastic Oscillator
    for window in [12, 24]:
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'stoch_k_{window}'] = stoch.stoch()
        df[f'stoch_d_{window}'] = stoch.stoch_signal()
    
    # Average True Range
    for window in [12, 24]:
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'atr_{window}'] = atr.average_true_range()
        df[f'atr_ratio_{window}'] = df[f'atr_{window}'] / df['Close']
    
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_span_a'] = ichimoku.ichimoku_a()
    df['ichimoku_span_b'] = ichimoku.ichimoku_b()
    
    # Volume indicators
    for window in [12, 24]:
        vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=window)
        df[f'vwap_{window}'] = vwap.volume_weighted_average_price()
        df[f'vwap_ratio_{window}'] = df['Close'] / df[f'vwap_{window}']
    
    # Accumulation/Distribution Index
    adi = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['adi'] = adi.acc_dist_index()
    
    # Donchian Channel
    for window in [12, 24]:
        donchian = DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'donchian_high_{window}'] = donchian.donchian_channel_hband()
        df[f'donchian_low_{window}'] = donchian.donchian_channel_lband()
        df[f'donchian_width_{window}'] = df[f'donchian_high_{window}'] - df[f'donchian_low_{window}']
    
    # True Strength Index
    for window in [12, 24]:
        tsi = TSIIndicator(close=df['Close'], window_slow=window, window_fast=window//2)
        df[f'tsi_{window}'] = tsi.tsi()
    
    # ADX (Average Directional Index)
    for window in [12, 24]:
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'adx_{window}'] = adx.adx()
    
    # CCI (Commodity Channel Index)
    for window in [12, 24]:
        cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'cci_{window}'] = cci.cci()
    
    # Daily return (even for intraday data)
    daily_return = DailyReturnIndicator(close=df['Close'])
    df['daily_return'] = daily_return.daily_return()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Price action features
    df['price_change'] = df['Close'] - df['Open']
    df['price_change_pct'] = df['price_change'] / df['Open']
    df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_ratio'] = df['Close'] / df['Open']
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_prediction_data(feature_df, prediction_horizon=4):  # 4*15min = 1 hour
    """
    Prepare data for predicting the next hour's price
    """
    # Create target (next hour's price)
    feature_df['target_price'] = feature_df['Close'].shift(-prediction_horizon)
    feature_df['target_return'] = feature_df['target_price'] / feature_df['Close'] - 1
    
    # Remove rows where target is NaN
    feature_df = feature_df.dropna(subset=['target_price'])
    
    # Separate features and target
    feature_columns = [col for col in feature_df.columns if col not in ['target_price', 'target_return', 'Close', 'Open', 'High', 'Low', 'Volume']]
    X = feature_df[feature_columns]
    y_price = feature_df['target_price']
    y_return = feature_df['target_return']
    
    return X, y_price, y_return, feature_df


def fetch_btc_data(period="60d", interval="15m"):
    """Fetch BTC historical price data with 15-minute intervals"""
    btc_data = yf.download("BTC-USD", period=period, interval=interval, progress=False)
    return btc_data
  
# Main execution
def create_btc_embedding(n_components=100):
    """
    Create a 100-dimensional embedding using PCA
    """
    print("Fetching BTC data...")
    btc_data = fetch_btc_data(period="60d", interval="15m")
    print(f"Fetched {len(btc_data)} records of BTC data")
    
    # Create comprehensive feature set
    print("Creating features...")
    feature_df = create_rolling_features(btc_data)
    print(f"Created {len(feature_df.columns)} features")
    
    # Prepare data for prediction
    # print("Preparing prediction data...")
    # X, y_price, y_return, feature_df = prepare_prediction_data(feature_df)
    X = feature_df
    print(f"Prepared {len(X)} samples with {len(X.columns)} features for prediction")
    
    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(X.tail(10))
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(X_scaled)

    embedding = np.tanh(embedding)
    
    # Create embedding DataFrame
    # embedding_df = pd.DataFrame(
    #     embedding,
    #     index=X.index,
    #     columns=[f'PC_{i+1}' for i in range(n_components)]
    # )
    
    # Print explained variance
    return embedding
