"""
analysis/market_regime.py — Deteksi Kondisi Pasar
Mendeteksi kondisi pasar: BULL_TREND, BEAR_TREND, SIDEWAYS, atau BREAKOUT.

Urutan evaluasi (prioritas):
  1. BREAKOUT  — BB squeeze + ekspansi + volume spike > 2x avg20
  2. BULL_TREND — ADX>25 AND +DI>-DI AND close>MA200
  3. BEAR_TREND — ADX>25 AND -DI>+DI AND close<MA200
  4. SIDEWAYS   — ADX<20 AND range sempit 20 hari (<5% harga)
  Default: SIDEWAYS
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import MarketRegime, MarketRegimeResult


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Indikator aktif per regime (Requirement 3.5 – 3.8)
_ACTIVE_INDICATORS: dict[MarketRegime, list[str]] = {
    MarketRegime.BULL_TREND: [
        "MA", "ADX", "RSI", "MACD", "OBV", "MFI",
        "BB", "Fibonacci", "Candlestick",
    ],
    MarketRegime.BEAR_TREND: [
        "MA", "ADX", "RSI", "MACD", "OBV", "MFI",
        "BB", "Fibonacci", "Candlestick",
    ],
    MarketRegime.SIDEWAYS: [
        "RSI", "Stochastic", "BB", "MFI", "CCI", "Williams%R",
    ],
    MarketRegime.BREAKOUT: [
        "BB", "Volume", "ADX", "MA", "RSI", "MACD", "OBV", "Candlestick",
    ],
}

# Deskripsi human-readable per regime
_DESCRIPTIONS: dict[MarketRegime, str] = {
    MarketRegime.BULL_TREND: (
        "Tren naik kuat: ADX>25, +DI>-DI, harga di atas MA200."
    ),
    MarketRegime.BEAR_TREND: (
        "Tren turun kuat: ADX>25, -DI>+DI, harga di bawah MA200."
    ),
    MarketRegime.SIDEWAYS: (
        "Pasar bergerak mendatar: ADX<20, range harga sempit 20 hari."
    ),
    MarketRegime.BREAKOUT: (
        "Breakout: BB squeeze diikuti ekspansi mendadak dan volume spike >2x rata-rata 20 hari."
    ),
}


# ---------------------------------------------------------------------------
# Helper — komputasi indikator inline
# ---------------------------------------------------------------------------

def _compute_adx(df: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
    """
    Hitung ADX, +DI, -DI secara inline dari kolom High, Low, Close.
    Mengembalikan (adx, plus_di, minus_di) dari candle terakhir.
    Mengembalikan (nan, nan, nan) jika data tidak cukup.
    """
    n = len(df)
    if n < period + 1:
        return float("nan"), float("nan"), float("nan")

    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )

    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing (RMA)
    def _rma(arr: np.ndarray, p: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        if len(arr) < p:
            return result
        result[p - 1] = arr[:p].sum()
        for i in range(p, len(arr)):
            result[i] = result[i - 1] - result[i - 1] / p + arr[i]
        return result

    atr_arr = _rma(tr, period)
    plus_dm_arr = _rma(plus_dm, period)
    minus_dm_arr = _rma(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di_arr = 100.0 * plus_dm_arr / atr_arr
        minus_di_arr = 100.0 * minus_dm_arr / atr_arr
        dx_arr = 100.0 * np.abs(plus_di_arr - minus_di_arr) / (plus_di_arr + minus_di_arr)

    adx_arr = _rma(np.nan_to_num(dx_arr, nan=0.0), period)

    # Nilai terakhir yang valid
    def _last_valid(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        return float(valid[-1]) if len(valid) > 0 else float("nan")

    return _last_valid(adx_arr), _last_valid(plus_di_arr), _last_valid(minus_di_arr)


def _compute_ma(series: pd.Series, period: int) -> float:
    """Hitung SMA periode terakhir. Mengembalikan nan jika data tidak cukup."""
    if len(series) < period:
        return float("nan")
    return float(series.iloc[-period:].mean())


def _compute_bb_bandwidth(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    """
    Hitung Bollinger Band Bandwidth (BBB) = (upper - lower) / middle * 100.
    Mengembalikan Series sepanjang df.
    """
    close = df["Close"]
    middle = close.rolling(period).mean()
    sigma = close.rolling(period).std(ddof=0)
    upper = middle + std * sigma
    lower = middle - std * sigma
    bbb = (upper - lower) / middle * 100.0
    return bbb


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------

def detect_market_regime(df: pd.DataFrame) -> MarketRegimeResult:
    """
    Deteksi kondisi pasar dari DataFrame OHLCV.

    Evaluasi 4 kondisi secara berurutan (prioritas tertinggi ke terendah):
      1. BREAKOUT
      2. BULL_TREND
      3. BEAR_TREND
      4. SIDEWAYS  (default jika tidak ada yang cocok)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom Open, High, Low, Close, Volume.
        Minimal 20 baris untuk analisis dasar; 200 baris untuk MA200.

    Returns
    -------
    MarketRegimeResult
    """
    if df is None or len(df) < 2:
        # Data tidak cukup — kembalikan SIDEWAYS sebagai default aman
        return MarketRegimeResult(
            regime=MarketRegime.SIDEWAYS,
            active_indicators=_ACTIVE_INDICATORS[MarketRegime.SIDEWAYS],
            description="Data tidak mencukupi untuk analisis regime.",
            adx_value=float("nan"),
            plus_di=float("nan"),
            minus_di=float("nan"),
        )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Ambil / hitung ADX, +DI, -DI
    # ------------------------------------------------------------------
    adx_val: float
    plus_di_val: float
    minus_di_val: float

    # Coba gunakan kolom pandas-ta jika sudah ada di df
    if "ADX_14" in df.columns and "DMP_14" in df.columns and "DMN_14" in df.columns:
        adx_val = float(df["ADX_14"].iloc[-1])
        plus_di_val = float(df["DMP_14"].iloc[-1])
        minus_di_val = float(df["DMN_14"].iloc[-1])
    else:
        adx_val, plus_di_val, minus_di_val = _compute_adx(df, period=14)

    # ------------------------------------------------------------------
    # 2. Hitung MA200 (opsional — skip jika data < 200 baris)
    # ------------------------------------------------------------------
    ma200: float
    if "SMA_200" in df.columns:
        ma200 = float(df["SMA_200"].iloc[-1])
    else:
        ma200 = _compute_ma(close, 200)  # nan jika < 200 baris

    current_close = float(close.iloc[-1])

    # ------------------------------------------------------------------
    # 3. Hitung Bollinger Band Bandwidth untuk BREAKOUT detection
    # ------------------------------------------------------------------
    # Cari kolom BBB secara fleksibel (BBB_20_2.0 atau BBB_20_2)
    bbb = None
    for bbb_col in ['BBB_20_2.0', 'BBB_20_2']:
        if bbb_col in df.columns:
            bbb = df[bbb_col]
            break
    if bbb is None:
        # Cari berdasarkan prefix
        for col in df.columns:
            if col.startswith('BBB_'):
                bbb = df[col]
                break
    if bbb is None:
        bbb = _compute_bb_bandwidth(df, period=20, std=2.0)

    # ------------------------------------------------------------------
    # 4. Evaluasi kondisi secara berurutan
    # ------------------------------------------------------------------

    # --- 4a. BREAKOUT (prioritas tertinggi) ---
    # Syarat:
    #   - BB squeeze: BBB saat ini < persentil ke-5 dari 100 candle terakhir
    #     (atau seluruh data jika < 100 candle)
    #   - BB expansion: BBB[-1] > BBB[-2] * 1.5
    #   - Volume spike: volume[-1] > 2x rata-rata volume 20 hari
    is_breakout = False
    if n >= 20:
        lookback = min(100, n)
        bbb_window = bbb.dropna().iloc[-lookback:]

        if len(bbb_window) >= 5:
            bbb_current = float(bbb.iloc[-1]) if not pd.isna(bbb.iloc[-1]) else float("nan")
            bbb_prev = float(bbb.iloc[-2]) if n >= 2 and not pd.isna(bbb.iloc[-2]) else float("nan")

            squeeze_threshold = float(np.percentile(bbb_window.values, 5))
            bb_squeeze = (not np.isnan(bbb_current)) and (bbb_current < squeeze_threshold)
            bb_expansion = (
                not np.isnan(bbb_current)
                and not np.isnan(bbb_prev)
                and bbb_prev > 0
                and bbb_current > bbb_prev * 1.5
            )

            vol_window = volume.iloc[-20:]
            avg_vol_20 = float(vol_window.mean())
            current_vol = float(volume.iloc[-1])
            volume_spike = avg_vol_20 > 0 and current_vol > 2.0 * avg_vol_20

            is_breakout = bb_squeeze and bb_expansion and volume_spike

    if is_breakout:
        regime = MarketRegime.BREAKOUT
        return MarketRegimeResult(
            regime=regime,
            active_indicators=_ACTIVE_INDICATORS[regime],
            description=_DESCRIPTIONS[regime],
            adx_value=adx_val,
            plus_di=plus_di_val,
            minus_di=minus_di_val,
        )

    # --- 4b. BULL_TREND ---
    # ADX>25 AND +DI>-DI AND close>MA200
    # Jika MA200 tidak tersedia (data < 200 baris), skip syarat MA200
    adx_ok = (not np.isnan(adx_val)) and adx_val > 25
    di_bull = (
        not np.isnan(plus_di_val)
        and not np.isnan(minus_di_val)
        and plus_di_val > minus_di_val
    )
    ma200_bull = np.isnan(ma200) or (current_close > ma200)  # skip jika tidak tersedia

    if adx_ok and di_bull and ma200_bull:
        regime = MarketRegime.BULL_TREND
        return MarketRegimeResult(
            regime=regime,
            active_indicators=_ACTIVE_INDICATORS[regime],
            description=_DESCRIPTIONS[regime],
            adx_value=adx_val,
            plus_di=plus_di_val,
            minus_di=minus_di_val,
        )

    # --- 4c. BEAR_TREND ---
    # ADX>25 AND -DI>+DI AND close<MA200
    di_bear = (
        not np.isnan(plus_di_val)
        and not np.isnan(minus_di_val)
        and minus_di_val > plus_di_val
    )
    ma200_bear = np.isnan(ma200) or (current_close < ma200)  # skip jika tidak tersedia

    if adx_ok and di_bear and ma200_bear:
        regime = MarketRegime.BEAR_TREND
        return MarketRegimeResult(
            regime=regime,
            active_indicators=_ACTIVE_INDICATORS[regime],
            description=_DESCRIPTIONS[regime],
            adx_value=adx_val,
            plus_di=plus_di_val,
            minus_di=minus_di_val,
        )

    # --- 4d. SIDEWAYS ---
    # ADX<20 AND range(high-low, 20 hari) < 5% dari harga saat ini
    adx_sideways = (not np.isnan(adx_val)) and adx_val < 20
    is_sideways = False
    if adx_sideways and n >= 20:
        recent_high = float(high.iloc[-20:].max())
        recent_low = float(low.iloc[-20:].min())
        price_range_pct = (recent_high - recent_low) / current_close * 100.0
        is_sideways = price_range_pct < 5.0

    if is_sideways:
        regime = MarketRegime.SIDEWAYS
        return MarketRegimeResult(
            regime=regime,
            active_indicators=_ACTIVE_INDICATORS[regime],
            description=_DESCRIPTIONS[regime],
            adx_value=adx_val,
            plus_di=plus_di_val,
            minus_di=minus_di_val,
        )

    # --- Default: SIDEWAYS ---
    return MarketRegimeResult(
        regime=MarketRegime.SIDEWAYS,
        active_indicators=_ACTIVE_INDICATORS[MarketRegime.SIDEWAYS],
        description="Kondisi pasar tidak memenuhi kriteria tren atau breakout — diklasifikasikan sebagai SIDEWAYS.",
        adx_value=adx_val,
        plus_di=plus_di_val,
        minus_di=minus_di_val,
    )


def get_active_indicators(regime: MarketRegime) -> list[str]:
    """
    Kembalikan daftar indikator aktif untuk regime pasar yang diberikan.

    Parameters
    ----------
    regime : MarketRegime
        Kondisi pasar yang terdeteksi.

    Returns
    -------
    list[str]
        Daftar nama indikator yang relevan untuk regime tersebut.
    """
    return list(_ACTIVE_INDICATORS.get(regime, _ACTIVE_INDICATORS[MarketRegime.SIDEWAYS]))
