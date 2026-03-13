import pandas as pd
import numpy as np
import requests
import time
import os
import math
import sys
from datetime import datetime

# ==========================================================
# CONFIGURACIÓN (adaptada a KuCoin)
# ==========================================================
ASSETS = ["BTC", "ETH", "SOL"]
TIMEFRAME = "3min"           # KuCoin usa "3min" en lugar de "3m"
LOOKBACK = 200
EMA_PERIOD = 20
ATR_PERIOD = 14
EDGE_THRESHOLD = 0.003
ATR_MULTIPLIER = 1.5
TP_EDGE = 0.30
LOOP_DELAY = 60               # segundos entre scans
REPORT_INTERVAL = 300         # cada 5 minutos
RETRIES = 3                   # reintentos por descarga fallida

# KuCoin API endpoint para velas
BASE_URL = "https://api.kucoin.com/api/v1/market/candles"

REPORT_DIR = "reports"
TRADES_FILE = os.path.join(REPORT_DIR, "trades.csv")
METRICS_FILE = os.path.join(REPORT_DIR, "metrics.json")

POSITION = None
metrics = {}                  # se cargará desde archivo o valores por defecto
LAST_REPORT = time.time()
SCAN_COUNTER = 0              # contador de escaneos

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def load_metrics():
    """Carga métricas previas si existen, si no usa valores por defecto."""
    global metrics
    if os.path.exists(METRICS_FILE):
        try:
            # Leer JSON como dict directamente (ya no como DataFrame)
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
            print(f"[{datetime.utcnow()}] 📂 Métricas cargadas: capital={metrics['capital']:.2f}, "
                  f"trades={metrics['trades']}, wins={metrics['wins']}, losses={metrics['losses']}")
        except Exception as e:
            print(f"[{datetime.utcnow()}] ⚠️ Error cargando métricas: {e}. Usando valores por defecto.")
            metrics = {"capital": 100.0, "trades": 0, "wins": 0, "losses": 0}
    else:
        metrics = {"capital": 100.0, "trades": 0, "wins": 0, "losses": 0}
        print(f"[{datetime.utcnow()}] 📂 No se encontraron métricas previas. Iniciando con capital 100.")

def save_metrics():
    """Guarda las métricas actuales como JSON (dict, no DataFrame)."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

def fetch(symbol):
    """
    Descarga velas de KuCoin con reintentos.
    Retorna DataFrame con columnas: time, open, high, low, close, volume.
    """
    # KuCoin espera símbolo con guión y tipo de vela en formato "3min", "5min", etc.
    params = {
        "symbol": f"{symbol}-USDT",
        "type": TIMEFRAME,
        "limit": LOOKBACK
    }
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"[{datetime.utcnow()}] 🌐 Descargando {symbol} (intento {attempt}/{RETRIES}) desde {BASE_URL}")
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "200000":
                raise Exception(f"Error KuCoin: {data.get('msg', 'desconocido')}")

            # La respuesta de KuCoin viene en una lista de listas:
            # [ [time, open, close, high, low, volume, turnover], ... ]
            candles = data["data"]
            df = pd.DataFrame(candles, columns=[
                "time", "open", "close", "high", "low", "volume", "turnover"
            ])
            # Convertir tipos numéricos y ordenar por tiempo (KuCoin devuelve descendente)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["time"] = pd.to_datetime(df["time"].astype(int), unit='s')
            df = df.sort_values("time")  # ascendente para cálculos
            print(f"[{datetime.utcnow()}] ✅ {len(df)} velas descargadas para {symbol}")
            return df
        except Exception as e:
            print(f"[{datetime.utcnow()}] ❌ Error descargando {symbol}: {e}")
            if attempt < RETRIES:
                time.sleep(2 ** attempt)  # backoff exponencial
            else:
                print(f"[{datetime.utcnow()}] ⚠️ No se pudo descargar {symbol} después de {RETRIES} intentos.")
                return None

def atr(df):
    """Calcula ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low,
                    abs(high - close.shift()),
                    abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()

def generate_signal(df, asset):
    """
    Genera señal si el edge supera el umbral.
    Devuelve dict con datos de la señal o None.
    """
    if df is None or df.empty:
        print(f"[{datetime.utcnow()}] ⚠️ No hay datos para {asset}, omitiendo.")
        return None

    ema = df["close"].ewm(span=EMA_PERIOD).mean()
    price = df["close"].iloc[-1]
    atr_val = atr(df).iloc[-1]
    ema_val = ema.iloc[-1]
    deviation = (price - ema_val) / ema_val
    edge = abs(deviation)

    print(f"[{datetime.utcnow()}] {asset} → precio={price:.2f}, EMA={ema_val:.2f}, "
          f"desviación={deviation:.5f}, edge={edge:.5f}")

    if edge < EDGE_THRESHOLD:
        print(f"[{datetime.utcnow()}] ⚠️ Edge por debajo del umbral ({EDGE_THRESHOLD}), sin señal.")
        return None

    tp_move = edge * TP_EDGE
    sl_move = (atr_val / price) * ATR_MULTIPLIER
    if deviation < 0:
        direction = "LONG"
        tp = price * (1 + tp_move)
        sl = price * (1 - sl_move)
    else:
        direction = "SHORT"
        tp = price * (1 - tp_move)
        sl = price * (1 + sl_move)

    print(f"[{datetime.utcnow()}] 💡 Señal {asset} {direction}: entry={price:.2f}, "
          f"TP={tp:.2f}, SL={sl:.2f}, edge={edge:.5f}")
    return {
        "asset": asset,
        "direction": direction,
        "entry": price,
        "tp": tp,
        "sl": sl,
        "edge": edge,
        "mfe": 0.0,
        "mae": 0.0
    }

def scan():
    """Escanea todos los activos y retorna la señal con mayor edge."""
    signals = []
    for asset in ASSETS:
        df = fetch(asset)
        signal = generate_signal(df, asset)
        if signal:
            signals.append(signal)

    if not signals:
        print(f"[{datetime.utcnow()}] 🔍 No se encontraron señales en este escaneo.")
        return None

    top_signal = max(signals, key=lambda x: x["edge"])
    print(f"[{datetime.utcnow()}] 🏆 Mejor señal: {top_signal['asset']} con edge={top_signal['edge']:.5f}")
    return top_signal

def check_position(pos):
    """
    Monitorea la posición abierta.
    Retorna (resultado, precio, sugerencia) donde resultado es "TP", "SL" o None.
    """
    df = fetch(pos["asset"])
    if df is None:
        # Si falla la descarga, no podemos verificar; asumimos que sigue abierta.
        return None, None, ""

    price = df["close"].iloc[-1]
    if pos["direction"] == "LONG":
        move = (price - pos["entry"]) / pos["entry"]
    else:
        move = (pos["entry"] - price) / pos["entry"]

    pos["mfe"] = max(pos["mfe"], move)
    pos["mae"] = min(pos["mae"], move)

    # Sugerencia simple (se puede mejorar)
    adjust = ""
    if move > pos["mfe"] * 0.7:
        adjust += "Considerar trailing SL | "
    if move < pos["mae"] * 0.7:
        adjust += "Considerar ajustar TP"

    # Comprobar TP/SL
    if pos["direction"] == "LONG":
        if price >= pos["tp"]:
            return "TP", price, adjust
        if price <= pos["sl"]:
            return "SL", price, adjust
    else:
        if price <= pos["tp"]:
            return "TP", price, adjust
        if price >= pos["sl"]:
            return "SL", price, adjust

    return None, price, adjust

def compute_metrics(trades_df):
    """Calcula MAE, MSE, RMSE sobre los PnL de los trades."""
    if trades_df.empty or len(trades_df) < 2:
        return {"MAE": 0.0, "MSE": 0.0, "RMSE": 0.0}
    errors = trades_df["pnl"] - trades_df["pnl"].mean()
    mae = errors.abs().mean()
    mse = (errors ** 2).mean()
    rmse = math.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}

def save_report():
    """Guarda un reporte de trades y métricas en un archivo de texto."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"report_{timestamp}.txt")

    trades = pd.read_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else pd.DataFrame()
    with open(report_file, "w") as f:
        f.write("=== TRADES ===\n")
        f.write(trades.to_string(index=False))
        f.write("\n\n=== MÉTRICAS GLOBALES ===\n")
        f.write(f"Capital: {metrics['capital']:.2f}\n")
        f.write(f"Trades: {metrics['trades']}\n")
        f.write(f"Wins: {metrics['wins']}\n")
        f.write(f"Losses: {metrics['losses']}\n")
        winrate = metrics['wins'] / metrics['trades'] if metrics['trades'] > 0 else 0
        f.write(f"Winrate: {winrate:.2%}\n")
        if not trades.empty:
            metric_vals = compute_metrics(trades)
            f.write(f"MAE: {metric_vals['MAE']:.5f}\n")
            f.write(f"MSE: {metric_vals['MSE']:.8f}\n")
            f.write(f"RMSE: {metric_vals['RMSE']:.5f}\n")

    print(f"[{datetime.utcnow()}] 📄 Reporte guardado: {report_file}")

def print_signal(s):
    """Muestra los detalles de una nueva señal de forma llamativa."""
    print("\n" + "=" * 60)
    print("🟢 NUEVA OPERACIÓN".center(60))
    print("=" * 60)
    print(f"Activo    : {s['asset']}/USDT")
    print(f"Dirección : {s['direction']}")
    print(f"Entrada   : {s['entry']:.4f}")
    print(f"TP        : {s['tp']:.4f}")
    print(f"SL        : {s['sl']:.4f}")
    print(f"Edge      : {s['edge']:.5f}")
    print("=" * 60)

# ==========================================================
# INICIALIZACIÓN
# ==========================================================
print(f"[{datetime.utcnow()}] 🚀 Iniciando bot de trading (KuCoin)")
print(f"[{datetime.utcnow()}] Verificando dependencias...")
print(f"[{datetime.utcnow()}] ✅ pandas {pd.__version__}")
print(f"[{datetime.utcnow()}] ✅ numpy {np.__version__}")
print(f"[{datetime.utcnow()}] ✅ requests {requests.__version__}")

# Cargar métricas previas
load_metrics()

print(f"[{datetime.utcnow()}] Configuración:")
print(f"  Activos: {ASSETS}")
print(f"  Timeframe: {TIMEFRAME}")
print(f"  EMA: {EMA_PERIOD} | ATR: {ATR_PERIOD}")
print(f"  Umbral edge: {EDGE_THRESHOLD}")
print(f"  Loop delay: {LOOP_DELAY}s | Reporte cada {REPORT_INTERVAL}s")
print(f"[{datetime.utcnow()}] ========== INICIO DEL LOOP PRINCIPAL ==========\n")

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================
while True:
    try:
        SCAN_COUNTER += 1
        print(f"[{datetime.utcnow()}] 🔍 Scan iniciado #{SCAN_COUNTER}")

        if POSITION is None:
            signal = scan()
            if signal:
                POSITION = signal
                print_signal(signal)
            else:
                print(f"[{datetime.utcnow()}] Sin señales en este ciclo.")
        else:
            result, price, adjust = check_position(POSITION)
            if result:  # TP o SL alcanzado
                # Calcular PnL
                entry = POSITION["entry"]
                if POSITION["direction"] == "LONG":
                    pnl = (price - entry) / entry
                else:
                    pnl = (entry - price) / entry

                # Actualizar métricas
                metrics["capital"] *= (1 + pnl)
                metrics["trades"] += 1
                if pnl > 0:
                    metrics["wins"] += 1
                else:
                    metrics["losses"] += 1
                winrate = metrics["wins"] / metrics["trades"] if metrics["trades"] > 0 else 0

                # Registrar trade
                trade_record = {
                    "time": datetime.utcnow().isoformat(),
                    **POSITION,
                    "exit": price,
                    "pnl": pnl,
                    "capital": metrics["capital"],
                    "winrate": winrate,
                    "adjust_suggestion": adjust.strip()
                }
                os.makedirs(REPORT_DIR, exist_ok=True)
                pd.DataFrame([trade_record]).to_csv(
                    TRADES_FILE, mode="a",
                    header=not os.path.exists(TRADES_FILE),
                    index=False
                )

                # Guardar métricas
                save_metrics()

                # Calcular métricas de error
                trades_df = pd.read_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else pd.DataFrame()
                metric_vals = compute_metrics(trades_df)

                print(f"\n[{datetime.utcnow()}] 🔴 TRADE CERRADO ({result})")
                print(f"   PnL: {pnl:.4%} | Capital: {metrics['capital']:.2f}")
                print(f"   Wins: {metrics['wins']} | Losses: {metrics['losses']} | Winrate: {winrate:.2%}")
                print(f"   Métricas de error: MAE={metric_vals['MAE']:.5f}, MSE={metric_vals['MSE']:.8f}, RMSE={metric_vals['RMSE']:.5f}")
                if adjust:
                    print(f"   Sugerencia: {adjust}")

                POSITION = None
            else:
                if price is not None:
                    print(f"[{datetime.utcnow()}] ℹ️ Trade activo: {POSITION['asset']} | "
                          f"Precio={price:.2f} | MFE={POSITION['mfe']:.5f} | MAE={POSITION['mae']:.5f} | {adjust}")
                else:
                    print(f"[{datetime.utcnow()}] ℹ️ Trade activo (sin datos de precio por error)")

        # Reporte periódico
        if time.time() - LAST_REPORT >= REPORT_INTERVAL:
            save_report()
            LAST_REPORT = time.time()

        print(f"[{datetime.utcnow()}] ⏸️ Esperando {LOOP_DELAY} segundos...\n")
        time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print(f"\n[{datetime.utcnow()}] 🛑 Bot detenido por el usuario.")
        save_report()
        save_metrics()
        break
    except Exception as e:
        print(f"[{datetime.utcnow()}] ❌ ERROR INESPERADO: {e}")
        # Opcional: guardar estado antes de continuar
        time.sleep(LOOP_DELAY)
