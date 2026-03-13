import pandas as pd
import numpy as np
import requests
import time, os, math
from datetime import datetime

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
ASSETS = ["BTC","ETH","SOL"]
TIMEFRAME = "3m"
LOOKBACK = 200
EMA_PERIOD = 20
ATR_PERIOD = 14
EDGE_THRESHOLD = 0.003
ATR_MULTIPLIER = 1.5
TP_EDGE = 0.30
LOOP_DELAY = 60            # segundos entre scans
REPORT_INTERVAL = 300      # cada 5 minutos
BASE_URL = "https://api.binance.com/api/v3/klines"

REPORT_DIR = "reports"
TRADES_FILE = os.path.join(REPORT_DIR,"trades.csv")
METRICS_FILE = os.path.join(REPORT_DIR,"metrics.json")

POSITION = None
metrics = {"capital":100,"trades":0,"wins":0,"losses":0}
LAST_REPORT = time.time()

# ==========================================================
# FUNCIONES
# ==========================================================
def fetch(symbol):
    """Descarga velas y muestra fuente"""
    print(f"[{datetime.utcnow()}] Fetching {symbol} from {BASE_URL}")
    params = {"symbol":f"{symbol}USDT","interval":TIMEFRAME,"limit":LOOKBACK}
    r = requests.get(BASE_URL, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    print(f"[{datetime.utcnow()}] {len(df)} candles fetched for {symbol}")
    return df

def atr(df):
    """Calcula ATR"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()

def generate_signal(df, asset):
    """Genera señal top-edge y explica por qué"""
    ema = df["close"].ewm(span=EMA_PERIOD).mean()
    price = df["close"].iloc[-1]
    atr_val = atr(df).iloc[-1]
    ema_val = ema.iloc[-1]
    deviation = (price-ema_val)/ema_val
    edge = abs(deviation)
    
    print(f"[{datetime.utcnow()}] {asset} deviation={deviation:.5f}, edge={edge:.5f}")
    
    if edge < EDGE_THRESHOLD:
        print(f"[{datetime.utcnow()}] Edge below threshold ({EDGE_THRESHOLD}), no signal generated")
        return None
    
    tp_move = edge*TP_EDGE
    sl_move = (atr_val/price)*ATR_MULTIPLIER
    if deviation < 0:
        direction = "LONG"
        tp = price*(1+tp_move)
        sl = price*(1-sl_move)
    else:
        direction = "SHORT"
        tp = price*(1-tp_move)
        sl = price*(1+sl_move)
    
    print(f"[{datetime.utcnow()}] Signal generated for {asset}: {direction}, TP={tp:.4f}, SL={sl:.4f}")
    
    return {
        "asset":asset, "direction":direction, "entry":price,
        "tp":tp, "sl":sl, "edge":edge, "mfe":0, "mae":0
    }

def scan():
    """Busca activo con mayor edge"""
    signals=[]
    for asset in ASSETS:
        df = fetch(asset)
        signal = generate_signal(df, asset)
        if signal:
            signals.append(signal)
    if not signals:
        print(f"[{datetime.utcnow()}] No signals found in this scan")
        return None
    top_signal = max(signals, key=lambda x:x["edge"])
    print(f"[{datetime.utcnow()}] Top signal selected: {top_signal['asset']} edge={top_signal['edge']:.5f}")
    return top_signal

def check_position(pos):
    """Revisa TP/SL y actualiza MFE/MAE"""
    df = fetch(pos["asset"])
    price = df["close"].iloc[-1]
    move = (price-pos["entry"])/pos["entry"] if pos["direction"]=="LONG" else (pos["entry"]-price)/pos["entry"]
    pos["mfe"] = max(pos["mfe"], move)
    pos["mae"] = min(pos["mae"], move)
    
    # Sugerencia de ajuste de SL/TP según métricas
    adjust = ""
    if move > pos["mfe"]*0.7:
        adjust += "Consider trailing SL | "
    if move < pos["mae"]*0.7:
        adjust += "Consider tightening TP"
    
    if pos["direction"]=="LONG":
        if price >= pos["tp"]: return "TP", price, adjust
        if price <= pos["sl"]: return "SL", price, adjust
    else:
        if price <= pos["tp"]: return "TP", price, adjust
        if price >= pos["sl"]: return "SL", price, adjust
    return None, price, adjust

def compute_metrics(trades_df):
    """Calcula MAE/MSE/RMSE global"""
    if trades_df.empty: return {"MAE":0,"MSE":0,"RMSE":0}
    errors = trades_df["pnl"] - trades_df["pnl"].mean()
    mae = errors.abs().mean()
    mse = (errors**2).mean()
    rmse = math.sqrt(mse)
    return {"MAE":mae,"MSE":mse,"RMSE":rmse}

def save_report():
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR,f"report_{timestamp}.txt")
    trades = pd.read_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else pd.DataFrame()
    metrics_current = pd.read_json(METRICS_FILE) if os.path.exists(METRICS_FILE) else metrics
    with open(report_file,"w") as f:
        f.write("=== TRADES ===\n")
        f.write(trades.to_string(index=False))
        f.write("\n\n=== METRICS ===\n")
        f.write(str(metrics_current))
    print(f"[{datetime.utcnow()}] Report saved: {report_file}")

def print_signal(s):
    print("\n"+"="*60)
    print("NEW TRADE")
    print("="*60)
    print(f"Asset : {s['asset']}/USDT")
    print(f"Direction : {s['direction']}")
    print(f"Entry : {s['entry']:.4f}")
    print(f"TP : {s['tp']:.4f}")
    print(f"SL : {s['sl']:.4f}")
    print(f"Edge : {s['edge']:.5f}")
    print("="*60)

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:
    try:
        if POSITION is None:
            signal = scan()
            if signal:
                POSITION = signal
                print_signal(signal)
            else:
                print(f"[{datetime.utcnow()}] No signal found this cycle")
        else:
            result, price, adjust = check_position(POSITION)
            if result:
                entry = POSITION["entry"]
                pnl = (price-entry)/entry if POSITION["direction"]=="LONG" else (entry-price)/entry
                metrics["capital"] *= 1+pnl
                metrics["trades"] += 1
                if pnl>0: metrics["wins"] += 1
                else: metrics["losses"] += 1
                trade_record = {
                    "time": datetime.utcnow(),
                    **POSITION,
                    "exit": price,
                    "pnl": pnl,
                    "capital": metrics["capital"],
                    "winrate": metrics["wins"]/metrics["trades"],
                    "adjust_suggestion": adjust
                }
                os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
                pd.DataFrame([trade_record]).to_csv(TRADES_FILE, mode="a", header=not os.path.exists(TRADES_FILE), index=False)
                pd.DataFrame([metrics]).to_json(METRICS_FILE)
                
                trades_df = pd.read_csv(TRADES_FILE)
                metric_vals = compute_metrics(trades_df)
                
                print(f"\n[{datetime.utcnow()}] TRADE CLOSED: {result} | PnL: {pnl:.4f} | Capital: {metrics['capital']:.2f}")
                print(f"Metrics: {metric_vals} | Winrate: {metrics['wins']/metrics['trades']:.2%}")
                if adjust:
                    print(f"Adjustment suggestion: {adjust}")
                
                POSITION = None
            else:
                print(f"[{datetime.utcnow()}] Trade running... Edge: {POSITION['edge']:.5f}, MFE: {POSITION['mfe']:.5f}, MAE: {POSITION['mae']:.5f}")

        if time.time()-LAST_REPORT >= REPORT_INTERVAL:
            save_report()
            LAST_REPORT = time.time()

        time.sleep(LOOP_DELAY)

    except Exception as e:
        print(f"[{datetime.utcnow()}] ERROR: {e}")
        time.sleep(LOOP_DELAY)
