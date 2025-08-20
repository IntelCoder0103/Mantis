import yfinance as yf
import pandas as pd
from prophet import Prophet
import requests
import numpy as np
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
import time
import secrets
import boto3
from botocore.client import Config
from timelock import Timelock
import json
import schedule
from btc import create_btc_embedding

from sklearn.preprocessing import StandardScaler

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
LOCK_TIME_SECONDS = 30

logger = logging.getLogger(__name__)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

normal_assets = ["BTC", "ETH", "EURUSD", "GBPUSD", "CADUSD", "NZDUSD", "CHFUSD", "XAUUSD", "XAGUSD"]
assets_symbol = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "CADUSD": "CADUSD=X",
    "NZDUSD": "NZDUSD=X",
    "CHFUSD": "CHFUSD=X",
    "XAUUSD": "GC=F",  # Gold
    "XAGUSD": "SI=F"   # Silver
}
# assets_symbol = ["BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "CADUSD=X", "NZDUSD=X", "CHFUSD=X", "GC=F", "SI=F"] #, "AAPL", "XAGUSD=X"]

currency_config =  {
    "samples": ["15m", "7d"],
    "multiplier": 0.5,
    "window_size": 96, # 1 day
    "dimentions": 2,
}
gold_config = {
    "samples": ["15m", "15d"],
    "multiplier": 1e-1,
    "window_size": 96 * 7, # 1 week
    "dimentions": 2,
}
silver_config = {
    "samples": ["15m", "15d"],
    "multiplier": 1e-1,
    "window_size": 96 * 7, # 1 week
    "dimentions": 2,
}
eth_config = {
    "samples": ["15m", "1mo"],
    "window_size": 96, # 1 day
    "dimentions": 2,
    "multiplier": 1e-1,
}
btc_config = {
    "samples": ["15m", "1mo"],
    "window_size": 300,
    "dimentions": 100,
    "multiplier": 1e-5,
}
config = {
    "BTC": btc_config,
    "ETH": eth_config,
    "EURUSD": currency_config,
    "GBPUSD": currency_config,
    "CADUSD": currency_config,
    "NZDUSD": currency_config,
    "CHFUSD": currency_config,
    "XAUUSD": gold_config,
    "XAGUSD": silver_config,
}

ACCOUNT_ID = "b88b18c28e9f44531976e51da0d5683a"
# API_TOKEN = "R8sAUdQhfNyQvbxrAXQ901TsKn1zy7zQYhkr7zJB"
ACCESS_KEY_ID = "5ef229aeb97dfc4d4b8bf014948db64e"
SECRET_ACCESS_KEY = "c581bea0e311749d282eaf07429d43800e7be83986d76c4773ee49da36767686"
BUCKET_NAME = "mantis"
# histories = {}

# for i, asset in enumerate(normal_assets):
#     asset_config = config[asset]
#     data_15mins = yf.download(f'{assets_symbol[i]}', interval=asset_config['interval'], period=asset_config['period'])  # 15-mins data
#     data_1h = yf.download(f'{assets_symbol[i]}', interval='1h', period='3mo')  # hourly data
#     data_1y = yf.download(f'{assets_symbol[i]}', interval='1d', period='1y')  # daily data
#     histories[asset] = {
#         '15mins': data_15mins,
#         '1h': data_1h,
#         '1y': data_1y
#     }

# latest_prices_url = "https://pub-ba8c1b8edb8046edaccecbd26b5ca7f8.r2.dev/latest_prices.json"
# Example Data {"timestamp": "2025-08-12T07:16:25.748811", "prices": {"BTC": 119094.8, "ETH": 4304, "EURUSD": 1.16112, "GBPUSD": 1.34436, "CADUSD": 0.72577, "NZDUSD": 0.593155, "CHFUSD": 1.2325248134635505, "XAUUSD": 3345.535, "XAGUSD": 37.87155}}
# Download latest prices from the provided URL
# response = requests.get(latest_prices_url, headers=headers)
# latest_prices_resp = pd.read_json(StringIO(response.text))
# Extract EUR/USD price from the latest prices
# eurusd_latest = latest_prices['prices']['EURUSD']
# latest_prices = latest_prices_resp['prices'].to_dict()
# timestamp = pd.to_datetime(latest_prices_resp['timestamp']['EURUSD'], utc=True).tz_localize(None)
# print(timestamp, eurusd_latest)


MY_HOTKEY = "5DJnNPMgkVEQZ2URiJPaQejK4rAw1D8koLt5VTdvbbFDrTHy"

# def training_data_for_btc():
#     asset_config = config['BTC']
#     data_1y = histories['BTC']['1y']
#     data_1y = data_1y[['Close']].reset_index()
#     data_1y.columns = ['ds', 'y']
    
#     def sigmoid_scale(x, center, width):
#         """Scale `x` to [-1, 1] using a sigmoid centered at `center` with scaling `width`."""
#         return 2 / (1 + np.exp(-(x - center)/width)) - 1

#     # Pick 50 15 mins data
#     # Pick 25 1 hour data
#     # Pick 20 daily data
#     # Pick 5 monthly data
    
#     data_15mins = yf.download('BTC-USD', interval='15m', period='1d')  # 15-mins data
#     data_1h = yf.download('BTC-USD', interval='1h', period='2d')  # hourly data
#     data_1d = yf.download('BTC-USD', interval='1d', period='30d')  # daily data
#     data_1mo = yf.download('BTC-USD', interval='1mo', period='1y')  # monthly data
    
#     data_15mins = data_15mins[['Close']].reset_index()
#     data_15mins.columns = ['ds', 'y']
#     data_1h = data_1h[['Close']].reset_index()
#     data_1h.columns = ['ds', 'y']
#     data_1d = data_1d[['Close']].reset_index()
#     data_1d.columns = ['ds', 'y']
#     data_1mo = data_1mo[['Close']].reset_index()
#     data_1mo.columns = ['ds', 'y']
    
#     data_center = np.median(data_1y['y'])
#     data_width = np.std(data_1y['y']) * 0.5  # Adjust width for desired steepness
    
#     all_data = np.concatenate((data_15mins['y'].values[:50], data_1h['y'].values[:25], data_1d['y'].values[:20], data_1mo['y'].values[:5]))
#     all_data = sigmoid_scale(all_data, data_center, data_width)
#     print(all_data)
    

# Helper functions needed for feature calculation
def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    return np.max(drawdown)

def calculate_atr_series(close, high, low, window=14):
    """Vectorized ATR for pandas Series"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_hurst_exponent(prices, max_lag=50):
    """Calculate Hurst Exponent to measure trend strength"""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def gen_samples_from_prices(prices, n_samples):
    prices = prices[::-1] # Reverse the array
    # prices = prices[1:]
    # print(prices)
    
    # Generate n_samples count samples in slideing window,
    price_series = []
    sample_len = len(prices) - (n_samples + 1) * 4
    for i in range(n_samples):
        sample = prices[i * 4 + sample_len:i * 4:-1]  # Reverse the sample
        price_series.append(sample)
    
    return price_series

def training_data_for_asset(asset:str):
    # data = histories[asset]['15mins']
    # data_1h = histories[asset]['1h']
    asset_config = config[asset]
    sample_config = asset_config['samples']
    window = asset_config['window_size']
    
    
    interval, period = sample_config
    data = yf.download(f'{assets_symbol[asset]}', interval=interval, period=period)
    # data_close = data[['Close']].reset_index()
    # data_high = data[['High']].reset_index()
    # data_low = data[['Low']].reset_index()
    # data_close.columns = ['ds', 'y']  # Prophet requires columns 'ds' (date) and 'y' (value)
    # data_high.columns = ['ds', 'y']
    # data_low.columns = ['ds', 'y']

    # # print(data.tail(20))
    # prices = data_close['y'].values
    # prices_high = data_high['y'].values
    # prices_low = data_low['y'].values

    # price_series = gen_samples_from_prices(prices, n_samples)
    # price_series_high = gen_samples_from_prices(prices_high, n_samples)
    # price_series_low = gen_samples_from_prices(prices_low, n_samples)
    
    df = {}
    
    for key, _ in data.columns:
        print(key)
        df[key] = data[key].to_numpy().flatten()
    df = pd.DataFrame(df, index = data.index)

    prices = df['Close']
    returns = prices.pct_change()
    rolling_returns = returns.rolling(window)

    features_df = pd.DataFrame({
        'total_return': (prices.rolling(window).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)),
        'volatility': rolling_returns.std(),
        'sharpe_ratio': rolling_returns.mean() / rolling_returns.std() * np.sqrt(252),
        'skewness': rolling_returns.skew(),
        'kurtosis': rolling_returns.kurt(),
        'current_price': prices,
        'prev_hour_price': prices.shift(5),
        'avg_true_range': calculate_atr_series(prices, df['High'], df['Low'], window=window),
        'max_drawdown': prices.rolling(window).apply(calculate_max_drawdown, raw=True),
        'hurst_exponent': prices.rolling(window).apply(calculate_hurst_exponent, raw=True),
        'volume': df['Volume'],
    })
    
    features_df = features_df.dropna()

    # features = []

    # for i, prices in enumerate(price_series):
    #     returns = np.diff(prices) / prices[:-1]  # Simple returns
        
    #     feature_set = {
    #         'total_return': (prices[-1] - prices[0]) / prices[0],
    #         'volatility': np.std(returns),
    #         'max_drawdown': calculate_max_drawdown(prices),
    #         'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),  # Annualized
    #         'avg_true_range': calculate_atr(prices, price_series_high[i], price_series_low[i], window=14),
    #         'hurst_exponent': calculate_hurst_exponent(prices),
    #         'skewness': pd.Series(returns).skew(),
    #         'kurtosis': pd.Series(returns).kurtosis(),
    #         'current_price': prices[-1],
    #         'prev_hour_price': prices[-5],
    #     }
    #     features.append(feature_set)

    # Convert to DataFrame
    # feature_df = pd.DataFrame(features)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    # print(x)

    # Step 1: Fit PCA on similar sequences (or just use it directly if you have one sample)
    pca = PCA(n_components=asset_config['dimentions'])
    x_pca = pca.fit_transform(scaled_features)
    x_pca *= asset_config['multiplier']  # Scale the PCA output by the multiplier

    # print(x_pca)
    x_tanh = np.tanh(x_pca)
    # print(x_tanh)
    # Step 2: Scale to [-1, 1]
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # x_scaled = scaler.fit_transform(x_pca)
    
    # print(x_tanh)
    return x_tanh[-1].tolist()

def encrypt_embeddings(embeddings_list):
    """Encrypt embeddings using timelock encryption as per miner guide"""
    try:
        logger.info("Starting timelock encryption process...")
        
        # Fetch beacon info to calculate a future round
        info_response = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10)
        if info_response.status_code != 200:
            logger.error(f"Failed to fetch beacon info: {info_response.status_code}")
            return embeddings_list
        
        info = info_response.json()
        future_time = time.time() + 30  # Target a round ~30 seconds in the future
        target_round = int((future_time - info["genesis_time"]) // info["period"])
        
        logger.info(f"Target round for encryption: {target_round}")
        
        # Create the plaintext by joining embeddings and the hotkey
        plaintext = f"{str(embeddings_list)}:::{MY_HOTKEY}"
        
        # Try to use the actual timelock library if available
        try:
            from timelock import Timelock
            logger.info("Using timelock library for encryption")
            
            # Handle different versions of the timelock library
            try:
                # Try the newer API first
                tlock = Timelock(DRAND_PUBLIC_KEY)
                salt = secrets.token_bytes(32)
                ciphertext_hex = tlock.tle(target_round, plaintext, salt).hex()
            except (AttributeError, TypeError):
                # Fallback for older versions - try different method names
                logger.info("Trying alternative timelock API for older version")
                try:
                    # Try alternative method names that might exist in older versions
                    tlock = Timelock()
                    salt = secrets.token_bytes(32)
                    # Try different possible method names
                    if hasattr(tlock, 'encrypt'):
                        ciphertext = tlock.encrypt(target_round, plaintext, salt)
                    elif hasattr(tlock, 'lock'):
                        ciphertext = tlock.lock(target_round, plaintext, salt)
                    else:
                        raise AttributeError("No suitable encryption method found")
                    
                    ciphertext_hex = ciphertext.hex() if hasattr(ciphertext, 'hex') else str(ciphertext)
                except Exception as alt_error:
                    logger.warning(f"Alternative timelock API failed: {alt_error}")
                    raise ImportError("Timelock library version incompatible")
            
            # Create encrypted payload in the exact format specified by miner guide
            encrypted_payload = {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
            logger.info("Timelock encryption completed successfully")
            return encrypted_payload
            
        except ImportError:
            logger.warning("Timelock library not available, using fallback encryption")
            # Fallback encryption method when timelock library is not available
            import hashlib
            salt = secrets.token_bytes(32)
            combined = plaintext.encode() + salt
            encrypted_hash = hashlib.sha256(combined).hexdigest()
            
            # Create encrypted payload with fallback method indicator
            encrypted_payload = {
                "round": target_round,
                "ciphertext": encrypted_hash,
                "salt": salt.hex(),
                "encryption_method": "sha256_fallback",
                "note": "Install 'pip install timelock==0.0.1.dev0' for proper timelock encryption"
            }
            
            logger.info("Fallback encryption completed successfully")
            return encrypted_payload
        
    except Exception as e:
        logger.error(f"Encryption failed: {str(e)}")
        logger.warning("Falling back to unencrypted output")
        return embeddings_list


def save_encrypted_payload(encrypted_payload, hotkey):
    """Save encrypted payload file as specified in miner guide"""
    try:
        # Create output directory if it doesn't exist
        import os
        os.makedirs("output", exist_ok=True)
        
        # Save encrypted payload with hotkey as filename (as per miner guide)
        payload_filename = hotkey
        payload_filepath = os.path.join("output", payload_filename)
        
        with open(payload_filepath, 'w') as f:
            json.dump(encrypted_payload, f, indent=2)
        
        logger.info(f"Encrypted payload saved to {payload_filepath}")
        return payload_filepath
        
    except Exception as e:
        logger.error(f"Error saving encrypted payload: {str(e)}")
        return None


def upload_r2_bucket(payload_filepath):
    ENDPOINT_URL = f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com'

    # Create the S3 client for R2
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )

    # Upload the file
    with open(payload_filepath, 'rb') as f:
        s3.upload_fileobj(f, BUCKET_NAME, MY_HOTKEY)

    print(f'✅ File uploaded successfully to https://{BUCKET_NAME}.{ACCOUNT_ID}.r2.dev/')

def save_results(filename="asset_predictions.json"):
    """Save prediction results to JSON file"""
    try:
        # Generate embeddings in the required format
        embeddings_list = generate_asset_embedding_dims()
        
        print(f"Generated embeddings: {embeddings_list}")
        
        # Encrypt the embeddings
        encrypted_payload = encrypt_embeddings(embeddings_list)
        
        # Save encrypted payload in miner guide format
        if isinstance(encrypted_payload, dict) and "ciphertext" in encrypted_payload:
            # This is the encrypted payload format
            payload_filepath = save_encrypted_payload(encrypted_payload, MY_HOTKEY)
            if payload_filepath:
                logger.info(f"Encrypted payload saved successfully to {payload_filepath}")
                upload_r2_bucket(payload_filepath)
            else:
                logger.error("Failed to save encrypted payload")
        else:
            # Fallback to unencrypted format
            logger.warning("Using unencrypted format due to encryption failure")
            encrypted_payload = embeddings_list
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs("output", exist_ok=True)
        
        # Save encrypted embeddings in the required format (list of lists)
        embeddings_filepath = os.path.join("output", filename)
        with open(embeddings_filepath, 'w') as f:
            json.dump(encrypted_payload, f, indent=2)
        
        logger.info(f"Embeddings saved to {embeddings_filepath}")

        return embeddings_filepath
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return None


# def training_data_for_asset(asset: str):
#     data_1y = histories[asset]['1y']
#     data_15mins = histories[asset]['15mins']
#     asset_config = config[asset]
    

#     def sigmoid_scale(x, center, width):
#         """Scale `x` to [-1, 1] using a sigmoid centered at `center` with scaling `width`."""
#         return 2 / (1 + np.exp(-(x - center)/width)) - 1

#     data_1y = data_1y[['Close']].reset_index()
#     data_1y.columns = ['ds', 'y']
#     data_center = np.median(data_1y['y'])
#     data_width = np.std(data_1y['y']) * 0.5  # Adjust width for desired steepness

#     def predict(data):
#         # Keep only the 'Close' price
#         data.index = data.index.tz_localize(None)  # Remove timezone
#         df = data[['Close']].reset_index()
#         df.columns = ['ds', 'y']  # Prophet requires columns 'ds' (date) and 'y' (value)


#         # Add the latest price as the last row
#         if asset in latest_prices and asset not in ['XAUUSD', 'XAGUSD']:
#             latest_row = pd.DataFrame({'ds': [timestamp], 'y': [latest_prices[asset]]})
#             df = pd.concat([df, latest_row], ignore_index=True)

#         current_price = latest_prices[asset]
#         center = data_center
#         width = data_width

#         # print(df.tail(20))
#         # Initialize and fit the model
#         model = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,  # Forex markets run 24/5, not strong weekly patterns
#             daily_seasonality=False,
#             changepoint_prior_scale=0.5  # Adjust sensitivity to trend changes
#         )

#         model.fit(df)

#         # current_time = pd.Timestamp.now(tz='UTC')
#         time_in_next_hour = timestamp + pd.Timedelta(hours=1)
#         time_in_next_hour = time_in_next_hour.tz_localize(None)
        
#         future = pd.DataFrame({'ds': time_in_next_hour}, index=[0])
#         forecast = model.predict(future)

#         # Get the predicted EUR/USD price
#         predicted_price = forecast['yhat'].iloc[0]
#         # Apply sigmoid scaling
#         normalized_current_price = sigmoid_scale(predicted_price, center, width)

#         # uncertainty = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
#         # MAX_UNCERTAINTY = asset_config['max_uncertainty']
#         # uncertainty_pct = min(MAX_UNCERTAINTY, (uncertainty / predicted_price)) # Percentage uncertainty

#         # # Normalize to [-1, 1]
#         # normalized_uncertainty = uncertainty_pct / MAX_UNCERTAINTY

#         print(f"Predicted {asset} on {time_in_next_hour}: {predicted_price:.4f}")
#         print(f"Normalized Price: {normalized_price:.4f}, Normalized Uncertainty: {normalized_uncertainty:.4f}")

#         return [normalized_price, normalized_uncertainty]


#     predict(data_15mins)

def generate_asset_embedding_dims():
    result = []
    # training_data_for_btc()
    for asset in normal_assets:
        print(f"Processing {asset}...")
        if asset == "BTC":
            # Run the embedding creation
            embedding = create_btc_embedding(n_components=100)
            result.append(embedding[-1].tolist())
            # print(embedding[-1])
        else:
            result.append(training_data_for_asset(asset))
        
    return result

if __name__ == "__main__":
    schedule.every(5).minutes.do(save_results)
    
    while True:
        schedule.run_pending()
        time.sleep(30)  # Sleep to avoid busy-waiting
