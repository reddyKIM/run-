import os, time, json, datetime as dt
from pathlib import Path
import requests
import pandas as pd

SYMBOL = os.getenv("SYMBOL", "ARBUSDT")
OUTDIR = Path(os.getenv("OUTDIR", "data"))
OUTDIR.mkdir(exist_ok=True, parents=True)

BINANCE_FAPI = "https://fapi.binance.com"

def log(*a): print("[BF]", *a, flush=True)

def session():
    s = requests.Session()
    s.headers.update({"User-Agent": "bf-collector/1.0"})
    return s
S = session()

def get_json(url, params=None, tries=5, backoff=0.6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=(5, 15))
            if r.status_code == 200:
                return r.json()
            else:
                log("HTTP", r.status_code, url, params)
        except Exception as e:
            log("ERR", e, url, params)
        time.sleep(backoff * (1 + i))
    return None

def server_time_ms():
    j = get_json(f"{BINANCE_FAPI}/fapi/v1/time")
    return int(j["serverTime"]) if j else int(time.time()*1000)

def to_iso(ms):
    return dt.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%dT%H:%M:%SZ")

def file_header_only(path: Path):
    if not path.exists(): return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            return len(f.readlines()) <= 1
    except Exception:
        return True

def ensure_csv(path: Path, df: pd.DataFrame):
    if df is None or df.empty: return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("wrote", path, len(df))
    return True

def fetch_klines(symbol, interval, start=None, end=None):
    rows = []
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1500}
    if start is not None: params["startTime"] = int(start)
    if end is not None: params["endTime"] = int(end)
    while True:
        j = get_json(url, params=params)
        if not j: break
        rows.extend(j)
        last_close = j[-1][6]  # closeTime
        nxt = int(last_close) + 1
        if end and nxt > end: break
        params["startTime"] = nxt
        if len(j) < 1500: break
        time.sleep(0.05)
    if not rows: return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts_utc"] = df["close_time"].apply(to_iso)
    return df

def backfill_01():
    fn = OUTDIR / "01_arb_15m_20d.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms()
    bars = 20*24*4
    start = now_ms - bars*15*60*1000
    df = fetch_klines(SYMBOL, "15m", start, now_ms)
    if df.empty: return
    df = df[["ts_utc","open","high","low","close","volume","quote_asset_volume","number_of_trades"]]\
        .rename(columns={"number_of_trades":"trades"})
    slot = pd.to_datetime(df["ts_utc"])
    sidx = (slot.dt.hour*60 + slot.dt.minute) // 15
    slot_avg = df.groupby(sidx)["volume"].mean()
    df["slot_avg_volume_20d"] = sidx.map(slot_avg)
    df["vol_ratio"] = (df["volume"] / df["slot_avg_volume_20d"]).round(4)
    ensure_csv(fn, df)

def backfill_02():
    fn = OUTDIR / "02_arb_1d_180d.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms()
    start = now_ms - 190*24*60*60*1000
    df = fetch_klines(SYMBOL, "1d", start, now_ms)
    if df.empty: return
    out = pd.DataFrame({
        "date_utc": df["ts_utc"],
        "o": df["open"].astype(float),
        "h": df["high"].astype(float),
        "l": df["low"].astype(float),
        "c": df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })
    lo = out["l"].tail(90).min(); hi = out["h"].tail(90).max(); rng = hi - lo
    out["fib_382"] = hi - 0.382*rng; out["fib_618"] = hi - 0.618*rng
    ensure_csv(fn, out)

def backfill_03():
    fn = OUTDIR / "03_arb_1m_today.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms()
    now_dt = dt.datetime.utcfromtimestamp(now_ms/1000)
    midnight = dt.datetime(now_dt.year, now_dt.month, now_dt.day)
    start = int(midnight.timestamp()*1000)
    df = fetch_klines(SYMBOL, "1m", start, now_ms)
    if df.empty: return
    out = pd.DataFrame({
        "ts_utc": df["ts_utc"],
        "o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"],
        "volume": df["volume"],
    })
    pv = (out["c"]*out["volume"]).cumsum()
    vv = out["volume"].cumsum()
    out["vwap"] = (pv/vv).round(6)
    out["above_vwap"] = (out["c"] >= out["vwap"]).astype(int)
    out["retest_flag"] = None
    ensure_csv(fn, out)

def backfill_04():
    fn = OUTDIR / "04_arb_funding_8h_30d.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms(); start = now_ms - 30*24*60*60*1000
    j = get_json(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                 {"symbol": SYMBOL, "startTime": start, "endTime": now_ms, "limit": 1000})
    if not j: return
    df = pd.DataFrame(j)
    df["funding_time_utc"] = df["fundingTime"].apply(to_iso)
    df["funding_realized_pct"] = pd.to_numeric(df["fundingRate"], errors="coerce")*100
    p = get_json(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": SYMBOL})
    df["funding_est_pct"] = float(p.get("predictedFundingRate", "0") or 0)*100 if p else None
    df["next_funding_ts"] = (to_iso(int(p["nextFundingTime"])) if p and p.get("nextFundingTime") else None)
    df["next_funding_ms"] = (int(p["nextFundingTime"]) if p and p.get("nextFundingTime") else None)
    df["symbol"] = SYMBOL
    keep = ["funding_time_utc","funding_realized_pct","funding_est_pct","next_funding_ts","next_funding_ms","symbol"]
    ensure_csv(fn, df[keep])

def backfill_05():
    fn = OUTDIR / "05_arb_oi_1h_30d.csv"
    if not file_header_only(fn): return
    j = get_json(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                 {"symbol": SYMBOL, "period":"1h", "limit":720, "contractType":"PERPETUAL"})
    if not j: return
    df = pd.DataFrame(j)
    df["ts_utc"] = df["timestamp"].apply(to_iso)
    df["binance_oi_contracts"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["binance_oi_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["oi_1h_pct"] = df["binance_oi_usd"].pct_change(1)*100
    df["oi_24h_pct"] = df["binance_oi_usd"].pct_change(24)*100
    df["bybit_oi_contracts"] = None
    df["total_oi_usd"] = df["binance_oi_usd"]
    keep = ["ts_utc","binance_oi_contracts","binance_oi_usd","bybit_oi_contracts","total_oi_usd","oi_1h_pct","oi_24h_pct"]
    ensure_csv(fn, df[keep])

def backfill_06():
    fn = OUTDIR / "06_arb_orderbook_depth_1min_8h.csv"
    if not file_header_only(fn): return
    j = get_json(f"{BINANCE_FAPI}/fapi/v1/depth", {"symbol": SYMBOL, "limit": 5000})
    if not j: return
    bids = [(float(p), float(q)) for p,q,_ in j.get("bids", [])[:5000]]
    asks = [(float(p), float(q)) for p,q,_ in j.get("asks", [])[:5000]]
    if not bids or not asks: return
    best_bid = bids[0][0]; best_ask = asks[0][0]
    mid = (best_bid + best_ask)/2.0
    spread_bps = (best_ask - best_bid)/mid*10000
    def sum_band(levels, is_bid, pct):
        total_qty = 0.0; total_usd = 0.0
        if is_bid:
            limit = mid*(1-pct)
            for price, qty in levels:
                if price >= limit: total_qty += qty; total_usd += price*qty
                else: break
        else:
            limit = mid*(1+pct)
            for price, qty in levels:
                if price <= limit: total_qty += qty; total_usd += price*qty
                else: break
        return total_qty, total_usd
    b05, b05u = sum_band(bids, True, 0.005); a05, a05u = sum_band(asks, False, 0.005)
    b10, b10u = sum_band(bids, True, 0.01);  a10, a10u = sum_band(asks, False, 0.01)
    out = pd.DataFrame([{
        "ts_utc": to_iso(server_time_ms()), "mid": mid, "spread_bps": spread_bps,
        "depth_bid_0p5pct": b05, "depth_ask_0p5pct": a05,
        "depth_bid_1pct": b10, "depth_ask_1pct": a10,
        "depth_bid_0p5pct_usd": b05u, "depth_ask_0p5pct_usd": a05u,
        "depth_bid_1pct_usd": b10u, "depth_ask_1pct_usd": a10u
    }])
    ensure_csv(fn, out)

def backfill_07():
    fn = OUTDIR / "07_arb_liquidations_1m_7d.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms(); start = now_ms - 7*24*60*60*1000
    all_rows = []; step = 24*60*60*1000
    t = start
    while t < now_ms:
        j = get_json(f"{BINANCE_FAPI}/fapi/v1/allForceOrders",
                     {"symbol": SYMBOL, "startTime": t, "endTime": min(t+step-1, now_ms), "limit": 1000})
        if j: all_rows.extend(j)
        t += step; time.sleep(0.2)
    if not all_rows: return
    df = pd.DataFrame(all_rows)
    df["qty"] = pd.to_numeric(df.get("executedQty", 0), errors="coerce")
    df["price"] = pd.to_numeric(df.get("avgPrice", 0), errors="coerce")
    df["quote"] = df["qty"]*df["price"]
    df["minute"] = (pd.to_datetime(df["time"], unit="ms").dt.floor("T"))
    df["bucket"] = df["side"].astype(str).str.upper().map({"BUY":"raw_buy_usd","SELL":"raw_sell_usd"})
    g = df.groupby(["minute","bucket"])["quote"].sum().unstack(fill_value=0)
    g["ts_utc"] = g.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    g["liq_long_usd"]  = g.get("raw_sell_usd", 0.0)  # SELL -> long liq
    g["liq_short_usd"] = g.get("raw_buy_usd", 0.0)   # BUY  -> short liq
    g["count_long"] = 0; g["count_short"] = 0
    keep = ["ts_utc","liq_long_usd","liq_short_usd","count_long","count_short","raw_sell_usd","raw_buy_usd"]
    ensure_csv(fn, g.reset_index(drop=True)[keep])

def backfill_08():
    fn = OUTDIR / "08_arb_basis_mark_index_1m_24h.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms(); start = now_ms - 24*60*60*1000
    def f_mark():
        j = get_json(f"{BINANCE_FAPI}/fapi/v1/markPriceKlines",
                     {"symbol": SYMBOL, "interval":"1m", "startTime": start, "endTime": now_ms, "limit": 1500})
        if not j: return None
        df = pd.DataFrame(j, columns=["open_time","open","high","low","close","ignore","close_time","i2","i3","i4","i5","i6"])
        df["ts_utc"] = df["close_time"].apply(to_iso)
        df["mark_price"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["ts_utc","mark_price"]]
    def f_index():
        j = get_json(f"{BINANCE_FAPI}/fapi/v1/indexPriceKlines",
                     {"pair": SYMBOL, "interval":"1m", "startTime": start, "endTime": now_ms, "limit": 1500})
        if not j: return None
        df = pd.DataFrame(j, columns=["open_time","open","high","low","close","ignore","close_time","i2","i3","i4","i5","i6"])
        df["ts_utc"] = df["close_time"].apply(to_iso)
        df["index_price"] = pd.to_numeric(df["close"], errors="coerce")
        return df[["ts_utc","index_price"]]
    dm = f_mark(); di = f_index()
    if dm is None or di is None: return
    df = pd.merge(dm, di, on="ts_utc", how="inner")
    df["basis_pct"] = (df["mark_price"]/df["index_price"] - 1.0)*100.0
    ensure_csv(fn, df)

def backfill_09():
    fn = OUTDIR / "09_arb_trades_imbalance_1m_24h.csv"
    if not file_header_only(fn): return
    now_ms = server_time_ms(); start = now_ms - 24*60*60*1000
    df = fetch_klines(SYMBOL, "1m", start, now_ms)
    if df.empty: return
    out = pd.DataFrame({
        "ts_utc": df["ts_utc"],
        "buy_qty": df["taker_buy_base"],
        "sell_qty": df["volume"] - df["taker_buy_base"],
        "buy_quote": df["taker_buy_quote"],
        "sell_quote": df["quote_asset_volume"] - df["taker_buy_quote"],
    })
    out["delta_quote"] = out["buy_quote"] - out["sell_quote"]
    ensure_csv(fn, out)

def main():
    backfill_01(); backfill_02(); backfill_03(); backfill_04(); backfill_05()
    backfill_06(); backfill_07(); backfill_08(); backfill_09()

if __name__ == "__main__":
    main()