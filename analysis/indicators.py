"""
analysis/indicators.py — Perhitungan Indikator Teknikal

Menghitung semua indikator teknikal menggunakan library `ta` (bukosabino/ta)
dengan fallback ke implementasi numpy murni jika `ta` tidak tersedia.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from models import DataAgeClassification

# Import ta dengan graceful fallback ke numpy
try:
    import ta as _ta_lib
    _TA_AVAILABLE = True
except ImportError:
    _ta_lib = None
    _TA_AVAILABLE = False
    logging.warning(
        "library 'ta' tidak tersedia — menggunakan implementasi numpy. "
        "Jalankan: pip install ta"
    )

from analysis.ipo_detector import get_ipo_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: nama indikator -> kolom output yang dihasilkan
# ---------------------------------------------------------------------------

_INDICATOR_COLUMNS: dict[str, list[str]] = {
    'MA20':        ['SMA_20'],
    'MA50':        ['SMA_50'],
    'MA200':       ['SMA_200'],
    'ADX':         ['ADX_14', 'DMP_14', 'DMN_14'],
    'RSI':         ['RSI_14'],
    'MACD':        ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'],
    'ROC':         ['ROC_10', 'ROC_20'],
    'WilliamsR':   ['WILLR_14'],
    'CCI':         ['CCI_20'],
    'BB':          ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0'],
    'ATR':         ['ATRr_14'],
    'OBV':         ['OBV'],
    'MFI':         ['MFI_14'],
    'Stochastic':  ['STOCHk_14_3_3', 'STOCHd_14_3_3'],
    'Candlestick': ['CDL_HAMMER', 'CDL_SHOOTING_STAR', 'CDL_DOJI_10_0.1', 'CDL_ENGULFING'],
}

# Indikator yang tidak ada di IPO config — aktif untuk STANDARD dan FULL
_ALWAYS_ON_FOR_STANDARD: list[str] = ['ROC', 'WilliamsR', 'CCI', 'ATR', 'Stochastic']


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------

def calculate_all_indicators(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> pd.DataFrame:
    """
    Hitung semua indikator teknikal dan tambahkan sebagai kolom ke df.
    Terapkan IPO config berdasarkan data_age.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV dengan kolom: Open, High, Low, Close, Volume.
        Index harus berupa DatetimeIndex.
    data_age : DataAgeClassification
        Klasifikasi usia data (IPO_NEW, IPO_PARTIAL, STANDARD, FULL).

    Returns
    -------
    pd.DataFrame
        df dengan kolom indikator tambahan. Indikator yang dinonaktifkan
        oleh IPO config akan diisi NaN.
    """
    if not _TA_AVAILABLE:
        logger.warning("library ta tidak tersedia — menggunakan implementasi numpy.")

    df = df.copy()
    n = len(df)

    # Ambil konfigurasi IPO
    ipo_config = get_ipo_config(data_age)

    # Tentukan apakah indikator "always on for standard" aktif
    standard_active = data_age in (DataAgeClassification.STANDARD, DataAgeClassification.FULL)

    # -----------------------------------------------------------------------
    # 8.2 Moving Averages (SMA 20, 50, 200)
    # -----------------------------------------------------------------------
    _calc_sma(df, ipo_config, n)

    # -----------------------------------------------------------------------
    # 8.4 ADX(14), +DI, -DI
    # -----------------------------------------------------------------------
    _calc_adx(df, ipo_config, n)

    # -----------------------------------------------------------------------
    # 8.5 RSI(14), MACD(12,26,9), ROC(10,20), Williams %R(14), CCI(20)
    # -----------------------------------------------------------------------
    _calc_rsi(df, ipo_config, n)
    _calc_macd(df, ipo_config, n)
    _calc_roc(df, standard_active, n)
    _calc_willr(df, standard_active, n)
    _calc_cci(df, standard_active, n)

    # -----------------------------------------------------------------------
    # 8.6 Bollinger Bands(20,2), ATR(14), OBV, MFI(14), Stochastic(14,3)
    # -----------------------------------------------------------------------
    _calc_bb(df, ipo_config, n)
    _calc_atr(df, standard_active, n)
    _calc_obv(df, ipo_config, n)
    _calc_mfi(df, ipo_config, n)
    _calc_stochastic(df, standard_active, n)

    # -----------------------------------------------------------------------
    # 8.7 Pola Candlestick
    # -----------------------------------------------------------------------
    _calc_candlestick(df, ipo_config, n)

    # -----------------------------------------------------------------------
    # 8.8 & 8.9 Terapkan IPO config dan tangani NaN
    # -----------------------------------------------------------------------
    _apply_ipo_config(df, ipo_config, standard_active)
    _log_nan_indicators(df, data_age)

    # Pastikan semua kolom indikator bertipe float (bukan int) agar NaN bisa disimpan
    _ensure_float_columns(df)

    return df


def _ensure_float_columns(df: pd.DataFrame) -> None:
    """
    Konversi semua kolom indikator ke float64 agar NaN dapat disimpan dengan benar.
    library ta kadang menghasilkan kolom bertipe int yang tidak bisa menyimpan NaN.
    """
    all_indicator_cols: list[str] = []
    for cols in _INDICATOR_COLUMNS.values():
        all_indicator_cols.extend(cols)

    for col in all_indicator_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                df[col] = pd.to_numeric(df[col], errors='coerce')


# ---------------------------------------------------------------------------
# Helper: get_disabled_indicators
# ---------------------------------------------------------------------------

def get_disabled_indicators(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> list[str]:
    """
    Return list of indicator names that are disabled (by IPO config) or
    have NaN values in the last row of df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame setelah calculate_all_indicators() dipanggil.
    data_age : DataAgeClassification
        Klasifikasi usia data.

    Returns
    -------
    list[str]
        Nama-nama indikator yang dinonaktifkan atau bernilai NaN.
    """
    ipo_config = get_ipo_config(data_age)
    standard_active = data_age in (DataAgeClassification.STANDARD, DataAgeClassification.FULL)
    disabled: list[str] = []

    for indicator, columns in _INDICATOR_COLUMNS.items():
        # Cek apakah dinonaktifkan oleh IPO config
        ipo_key = _get_ipo_key(indicator)
        if ipo_key is not None:
            if not ipo_config.get(ipo_key, True):
                disabled.append(indicator)
                continue
        elif indicator in _ALWAYS_ON_FOR_STANDARD:
            if not standard_active:
                disabled.append(indicator)
                continue

        # Cek apakah kolom ada dan bernilai NaN di baris terakhir
        if df.empty:
            disabled.append(indicator)
            continue

        last_row = df.iloc[-1]
        for col in columns:
            if col not in df.columns or pd.isna(last_row.get(col, np.nan)):
                if indicator not in disabled:
                    disabled.append(indicator)
                break

    return disabled


def _get_ipo_key(indicator: str) -> Optional[str]:
    """Kembalikan kunci IPO config untuk nama indikator internal."""
    mapping = {
        'MA20':        'MA20',
        'MA50':        'MA50',
        'MA200':       'MA200',
        'ADX':         'ADX',
        'RSI':         'RSI',
        'MACD':        'MACD',
        'BB':          'BB',
        'OBV':         'OBV',
        'MFI':         'MFI',
        'Candlestick': 'Candlestick',
    }
    return mapping.get(indicator)


# ---------------------------------------------------------------------------
# Private helpers: perhitungan per indikator
# ---------------------------------------------------------------------------

def _set_nan_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Set kolom-kolom ke NaN (dinonaktifkan)."""
    for col in columns:
        df[col] = np.nan


# ---------------------------------------------------------------------------
# Implementasi numpy fallback untuk semua indikator
# ---------------------------------------------------------------------------

def _np_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average menggunakan numpy."""
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1: i + 1])
    return result


def _np_ema(arr: np.ndarray, window: int) -> np.ndarray:
    """Exponential Moving Average menggunakan numpy."""
    result = np.full(len(arr), np.nan)
    k = 2.0 / (window + 1)
    # Cari indeks pertama yang valid
    start = window - 1
    if start >= len(arr):
        return result
    result[start] = np.mean(arr[:window])
    for i in range(start + 1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def _np_rsi(arr: np.ndarray, window: int = 14) -> np.ndarray:
    """Relative Strength Index menggunakan numpy (metode Wilder/EMA)."""
    result = np.full(len(arr), np.nan)
    if len(arr) < window + 1:
        return result
    delta = np.diff(arr.astype(float))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Rata-rata awal
    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])
    for i in range(window, len(arr)):
        idx = i - 1  # indeks di delta
        avg_gain = (avg_gain * (window - 1) + gain[idx]) / window
        avg_loss = (avg_loss * (window - 1) + loss[idx]) / window
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def _np_macd(
    arr: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram menggunakan numpy EMA."""
    ema_fast = _np_ema(arr, fast)
    ema_slow = _np_ema(arr, slow)
    macd_line = ema_fast - ema_slow
    # Signal: EMA dari MACD line (hanya pada nilai yang valid)
    signal_line = np.full(len(arr), np.nan)
    valid_mask = ~np.isnan(macd_line)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= signal:
        start_idx = valid_indices[signal - 1]
        signal_vals = _np_ema(macd_line[valid_indices], signal)
        for j, vi in enumerate(valid_indices):
            if vi >= start_idx:
                signal_line[vi] = signal_vals[j]
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _np_bb(
    arr: np.ndarray, window: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: lower, middle, upper, bandwidth menggunakan numpy."""
    n = len(arr)
    bbl = np.full(n, np.nan)
    bbm = np.full(n, np.nan)
    bbu = np.full(n, np.nan)
    bbb = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_data = arr[i - window + 1: i + 1]
        mean = np.mean(window_data)
        std = np.std(window_data, ddof=0)
        bbm[i] = mean
        bbl[i] = mean - num_std * std
        bbu[i] = mean + num_std * std
        if mean != 0:
            bbb[i] = (bbu[i] - bbl[i]) / mean * 100
    return bbl, bbm, bbu, bbb


def _np_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> np.ndarray:
    """Average True Range menggunakan numpy (metode Wilder)."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < 2:
        return result
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    if n < window:
        return result
    result[window - 1] = np.mean(tr[:window])
    for i in range(window, n):
        result[i] = (result[i - 1] * (window - 1) + tr[i]) / window
    return result


def _np_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, -DI menggunakan numpy (metode Wilder)."""
    n = len(close)
    adx = np.full(n, np.nan)
    dmp = np.full(n, np.nan)
    dmn = np.full(n, np.nan)
    if n < window * 2:
        return adx, dmp, dmn

    tr = np.full(n, np.nan)
    dm_plus = np.full(n, np.nan)
    dm_minus = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    dm_plus[0] = 0.0
    dm_minus[0] = 0.0
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        dm_plus[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Wilder smoothing
    atr_w = np.full(n, np.nan)
    dmp_w = np.full(n, np.nan)
    dmn_w = np.full(n, np.nan)
    atr_w[window] = np.sum(tr[1: window + 1])
    dmp_w[window] = np.sum(dm_plus[1: window + 1])
    dmn_w[window] = np.sum(dm_minus[1: window + 1])
    for i in range(window + 1, n):
        atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / window + tr[i]
        dmp_w[i] = dmp_w[i - 1] - dmp_w[i - 1] / window + dm_plus[i]
        dmn_w[i] = dmn_w[i - 1] - dmn_w[i - 1] / window + dm_minus[i]

    for i in range(window, n):
        if atr_w[i] != 0:
            dmp[i] = 100.0 * dmp_w[i] / atr_w[i]
            dmn[i] = 100.0 * dmn_w[i] / atr_w[i]

    # DX dan ADX
    dx = np.full(n, np.nan)
    for i in range(window, n):
        di_sum = dmp[i] + dmn[i]
        if di_sum != 0:
            dx[i] = 100.0 * abs(dmp[i] - dmn[i]) / di_sum

    adx[window * 2 - 1] = np.nanmean(dx[window: window * 2])
    for i in range(window * 2, n):
        if not np.isnan(adx[i - 1]) and not np.isnan(dx[i]):
            adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window

    return adx, dmp, dmn


def _np_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume menggunakan numpy."""
    n = len(close)
    result = np.full(n, np.nan)
    if n == 0:
        return result
    result[0] = float(volume[0])
    for i in range(1, n):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result


def _np_mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Money Flow Index menggunakan numpy."""
    n = len(close)
    result = np.full(n, np.nan)
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    for i in range(window, n):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - window + 1, i + 1):
            if typical_price[j] > typical_price[j - 1]:
                pos_flow += raw_money_flow[j]
            elif typical_price[j] < typical_price[j - 1]:
                neg_flow += raw_money_flow[j]
        if neg_flow == 0:
            result[i] = 100.0
        else:
            mfr = pos_flow / neg_flow
            result[i] = 100.0 - (100.0 / (1.0 + mfr))
    return result


def _np_stoch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic %K dan %D menggunakan numpy."""
    n = len(close)
    stoch_k = np.full(n, np.nan)
    for i in range(k_window - 1, n):
        highest_high = np.max(high[i - k_window + 1: i + 1])
        lowest_low = np.min(low[i - k_window + 1: i + 1])
        denom = highest_high - lowest_low
        if denom != 0:
            stoch_k[i] = 100.0 * (close[i] - lowest_low) / denom
    stoch_d = _np_sma(stoch_k, d_window)
    return stoch_k, stoch_d


def _np_willr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> np.ndarray:
    """Williams %R menggunakan numpy."""
    n = len(close)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        highest_high = np.max(high[i - window + 1: i + 1])
        lowest_low = np.min(low[i - window + 1: i + 1])
        denom = highest_high - lowest_low
        if denom != 0:
            result[i] = -100.0 * (highest_high - close[i]) / denom
    return result


def _np_cci(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20
) -> np.ndarray:
    """Commodity Channel Index menggunakan numpy."""
    n = len(close)
    result = np.full(n, np.nan)
    typical_price = (high + low + close) / 3.0
    for i in range(window - 1, n):
        tp_window = typical_price[i - window + 1: i + 1]
        mean_tp = np.mean(tp_window)
        mean_dev = np.mean(np.abs(tp_window - mean_tp))
        if mean_dev != 0:
            result[i] = (typical_price[i] - mean_tp) / (0.015 * mean_dev)
    return result


def _np_roc(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """Rate of Change menggunakan numpy."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window, n):
        if arr[i - window] != 0:
            result[i] = ((arr[i] - arr[i - window]) / arr[i - window]) * 100.0
    return result


def _calc_sma(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung SMA 20, 50, 200."""
    close = df['Close']

    # SMA_20
    if ipo_config.get('MA20', False) and n >= 20:
        try:
            if _TA_AVAILABLE:
                df['SMA_20'] = _ta_lib.trend.sma_indicator(close, window=20)
            else:
                df['SMA_20'] = _np_sma(close.values, 20)
        except Exception as e:
            logger.warning(f"Gagal menghitung SMA_20: {e}")
            df['SMA_20'] = np.nan
    else:
        df['SMA_20'] = np.nan

    # SMA_50
    if ipo_config.get('MA50', False) and n >= 50:
        try:
            if _TA_AVAILABLE:
                df['SMA_50'] = _ta_lib.trend.sma_indicator(close, window=50)
            else:
                df['SMA_50'] = _np_sma(close.values, 50)
        except Exception as e:
            logger.warning(f"Gagal menghitung SMA_50: {e}")
            df['SMA_50'] = np.nan
    else:
        df['SMA_50'] = np.nan

    # SMA_200
    if ipo_config.get('MA200', False) and n >= 200:
        try:
            if _TA_AVAILABLE:
                df['SMA_200'] = _ta_lib.trend.sma_indicator(close, window=200)
            else:
                df['SMA_200'] = _np_sma(close.values, 200)
        except Exception as e:
            logger.warning(f"Gagal menghitung SMA_200: {e}")
            df['SMA_200'] = np.nan
    else:
        df['SMA_200'] = np.nan


def _calc_adx(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung ADX(14), +DI (DMP_14), -DI (DMN_14)."""
    adx_cols = _INDICATOR_COLUMNS['ADX']

    if ipo_config.get('ADX', False) and n >= 28:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            if _TA_AVAILABLE:
                df['ADX_14'] = _ta_lib.trend.adx(high, low, close, window=14)
                df['DMP_14'] = _ta_lib.trend.adx_pos(high, low, close, window=14)
                df['DMN_14'] = _ta_lib.trend.adx_neg(high, low, close, window=14)
            else:
                adx_val, dmp_val, dmn_val = _np_adx(
                    high.values, low.values, close.values, window=14
                )
                df['ADX_14'] = adx_val
                df['DMP_14'] = dmp_val
                df['DMN_14'] = dmn_val
        except Exception as e:
            logger.warning(f"Gagal menghitung ADX: {e}")
            _set_nan_columns(df, adx_cols)
    else:
        _set_nan_columns(df, adx_cols)


def _calc_rsi(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung RSI(14)."""
    if ipo_config.get('RSI', False) and n >= 14:
        try:
            close = df['Close']
            if _TA_AVAILABLE:
                df['RSI_14'] = _ta_lib.momentum.rsi(close, window=14)
            else:
                df['RSI_14'] = _np_rsi(close.values, window=14)
        except Exception as e:
            logger.warning(f"Gagal menghitung RSI: {e}")
            df['RSI_14'] = np.nan
    else:
        df['RSI_14'] = np.nan


def _calc_macd(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung MACD(12,26,9)."""
    macd_cols = _INDICATOR_COLUMNS['MACD']

    if ipo_config.get('MACD', False) and n >= 35:
        try:
            close = df['Close']
            if _TA_AVAILABLE:
                df['MACD_12_26_9'] = _ta_lib.trend.macd(
                    close, window_slow=26, window_fast=12
                )
                df['MACDs_12_26_9'] = _ta_lib.trend.macd_signal(
                    close, window_slow=26, window_fast=12, window_sign=9
                )
                df['MACDh_12_26_9'] = _ta_lib.trend.macd_diff(
                    close, window_slow=26, window_fast=12, window_sign=9
                )
            else:
                macd_line, signal_line, histogram = _np_macd(
                    close.values, fast=12, slow=26, signal=9
                )
                df['MACD_12_26_9'] = macd_line
                df['MACDs_12_26_9'] = signal_line
                df['MACDh_12_26_9'] = histogram
        except Exception as e:
            logger.warning(f"Gagal menghitung MACD: {e}")
            _set_nan_columns(df, macd_cols)
    else:
        _set_nan_columns(df, macd_cols)


def _calc_roc(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung ROC(10) dan ROC(20)."""
    if standard_active and n >= 20:
        try:
            close = df['Close']
            if _TA_AVAILABLE:
                df['ROC_10'] = _ta_lib.momentum.roc(close, window=10)
                df['ROC_20'] = _ta_lib.momentum.roc(close, window=20)
            else:
                df['ROC_10'] = _np_roc(close.values, window=10)
                df['ROC_20'] = _np_roc(close.values, window=20)
        except Exception as e:
            logger.warning(f"Gagal menghitung ROC: {e}")
            df['ROC_10'] = np.nan
            df['ROC_20'] = np.nan
    else:
        df['ROC_10'] = np.nan
        df['ROC_20'] = np.nan


def _calc_willr(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung Williams %R(14)."""
    if standard_active and n >= 14:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            if _TA_AVAILABLE:
                df['WILLR_14'] = _ta_lib.momentum.williams_r(high, low, close, lbp=14)
            else:
                df['WILLR_14'] = _np_willr(high.values, low.values, close.values, window=14)
        except Exception as e:
            logger.warning(f"Gagal menghitung Williams %R: {e}")
            df['WILLR_14'] = np.nan
    else:
        df['WILLR_14'] = np.nan


def _calc_cci(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung CCI(20)."""
    if standard_active and n >= 20:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            if _TA_AVAILABLE:
                df['CCI_20'] = _ta_lib.trend.cci(high, low, close, window=20)
            else:
                df['CCI_20'] = _np_cci(high.values, low.values, close.values, window=20)
        except Exception as e:
            logger.warning(f"Gagal menghitung CCI: {e}")
            df['CCI_20'] = np.nan
    else:
        df['CCI_20'] = np.nan


def _calc_bb(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung Bollinger Bands(20, 2)."""
    bb_cols = _INDICATOR_COLUMNS['BB']

    if ipo_config.get('BB', False) and n >= 20:
        try:
            close = df['Close']
            if _TA_AVAILABLE:
                df['BBL_20_2.0'] = _ta_lib.volatility.bollinger_lband(
                    close, window=20, window_dev=2
                )
                df['BBM_20_2.0'] = _ta_lib.volatility.bollinger_mavg(close, window=20)
                df['BBU_20_2.0'] = _ta_lib.volatility.bollinger_hband(
                    close, window=20, window_dev=2
                )
                df['BBB_20_2.0'] = _ta_lib.volatility.bollinger_wband(
                    close, window=20, window_dev=2
                )
            else:
                bbl, bbm, bbu, bbb = _np_bb(close.values, window=20, num_std=2)
                df['BBL_20_2.0'] = bbl
                df['BBM_20_2.0'] = bbm
                df['BBU_20_2.0'] = bbu
                df['BBB_20_2.0'] = bbb
        except Exception as e:
            logger.warning(f"Gagal menghitung Bollinger Bands: {e}")
            _set_nan_columns(df, bb_cols)
    else:
        _set_nan_columns(df, bb_cols)


def _calc_atr(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung ATR(14)."""
    if standard_active and n >= 14:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            if _TA_AVAILABLE:
                df['ATRr_14'] = _ta_lib.volatility.average_true_range(
                    high, low, close, window=14
                )
            else:
                df['ATRr_14'] = _np_atr(high.values, low.values, close.values, window=14)
        except Exception as e:
            logger.warning(f"Gagal menghitung ATR: {e}")
            df['ATRr_14'] = np.nan
    else:
        df['ATRr_14'] = np.nan


def _calc_obv(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung OBV."""
    if ipo_config.get('OBV', False) and n >= 2:
        try:
            close = df['Close']
            volume = df['Volume']
            if _TA_AVAILABLE:
                df['OBV'] = _ta_lib.volume.on_balance_volume(close, volume)
            else:
                df['OBV'] = _np_obv(close.values, volume.values)
        except Exception as e:
            logger.warning(f"Gagal menghitung OBV: {e}")
            df['OBV'] = np.nan
    else:
        df['OBV'] = np.nan


def _calc_mfi(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung MFI(14)."""
    if ipo_config.get('MFI', False) and n >= 14:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            volume = df['Volume']
            if _TA_AVAILABLE:
                df['MFI_14'] = _ta_lib.volume.money_flow_index(
                    high, low, close, volume, window=14
                )
            else:
                df['MFI_14'] = _np_mfi(
                    high.values, low.values, close.values, volume.values, window=14
                )
        except Exception as e:
            logger.warning(f"Gagal menghitung MFI: {e}")
            df['MFI_14'] = np.nan
    else:
        df['MFI_14'] = np.nan


def _calc_stochastic(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung Stochastic(14, 3)."""
    stoch_cols = _INDICATOR_COLUMNS['Stochastic']

    if standard_active and n >= 17:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            if _TA_AVAILABLE:
                df['STOCHk_14_3_3'] = _ta_lib.momentum.stoch(
                    high, low, close, window=14, smooth_window=3
                )
                df['STOCHd_14_3_3'] = _ta_lib.momentum.stoch_signal(
                    high, low, close, window=14, smooth_window=3
                )
            else:
                stoch_k, stoch_d = _np_stoch(
                    high.values, low.values, close.values, k_window=14, d_window=3
                )
                df['STOCHk_14_3_3'] = stoch_k
                df['STOCHd_14_3_3'] = stoch_d
        except Exception as e:
            logger.warning(f"Gagal menghitung Stochastic: {e}")
            _set_nan_columns(df, stoch_cols)
    else:
        _set_nan_columns(df, stoch_cols)


def _calc_candlestick(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """
    Hitung pola candlestick secara manual menggunakan numpy.
    Library ta tidak menyediakan candlestick patterns, sehingga
    implementasi ini selalu menggunakan numpy.
    """
    cdl_cols = _INDICATOR_COLUMNS['Candlestick']

    if ipo_config.get('Candlestick', False) and n >= 3:
        try:
            op = df['Open'].values
            hi = df['High'].values
            lo = df['Low'].values
            cl = df['Close'].values

            # Hammer: body kecil di atas, lower shadow panjang (>= 2x body), upper shadow kecil
            body = np.abs(cl - op)
            lower_shadow = np.where(cl >= op, op - lo, cl - lo)
            upper_shadow = np.where(cl >= op, hi - cl, hi - op)
            avg_body = np.where(body > 0, body, 1e-10)
            hammer = (
                (lower_shadow >= 2.0 * avg_body) &
                (upper_shadow <= 0.3 * avg_body) &
                (body > 0)
            ).astype(float)
            hammer[hammer == 0] = np.nan

            # Shooting Star: body kecil di bawah, upper shadow panjang (>= 2x body), lower shadow kecil
            shooting_star = (
                (upper_shadow >= 2.0 * avg_body) &
                (lower_shadow <= 0.3 * avg_body) &
                (body > 0)
            ).astype(float)
            shooting_star[shooting_star == 0] = np.nan

            # Doji: body sangat kecil relatif terhadap range (< 10% dari range)
            candle_range = hi - lo
            candle_range = np.where(candle_range > 0, candle_range, 1e-10)
            doji = (body / candle_range < 0.1).astype(float)
            doji[doji == 0] = np.nan

            # Engulfing: candle saat ini menelan candle sebelumnya
            engulfing = np.full(n, np.nan)
            for i in range(1, n):
                prev_body_lo = min(op[i - 1], cl[i - 1])
                prev_body_hi = max(op[i - 1], cl[i - 1])
                curr_body_lo = min(op[i], cl[i])
                curr_body_hi = max(op[i], cl[i])
                if curr_body_lo < prev_body_lo and curr_body_hi > prev_body_hi:
                    # Bullish engulfing: candle saat ini bullish, sebelumnya bearish
                    if cl[i] > op[i] and cl[i - 1] < op[i - 1]:
                        engulfing[i] = 1.0
                    # Bearish engulfing: candle saat ini bearish, sebelumnya bullish
                    elif cl[i] < op[i] and cl[i - 1] > op[i - 1]:
                        engulfing[i] = -1.0

            df['CDL_HAMMER'] = hammer
            df['CDL_SHOOTING_STAR'] = shooting_star
            df['CDL_DOJI_10_0.1'] = doji
            df['CDL_ENGULFING'] = engulfing

        except Exception as e:
            logger.warning(f"Gagal menghitung pola candlestick: {e}")
            _set_nan_columns(df, cdl_cols)
    else:
        _set_nan_columns(df, cdl_cols)


# ---------------------------------------------------------------------------
# 8.8 Terapkan IPO config: disable indikator yang tidak tersedia
# ---------------------------------------------------------------------------

def _apply_ipo_config(
    df: pd.DataFrame,
    ipo_config: dict,
    standard_active: bool,
) -> None:
    """
    Pastikan semua kolom indikator yang dinonaktifkan oleh IPO config
    benar-benar bernilai NaN (override jika ada nilai yang terhitung).
    """
    # Indikator yang dikontrol oleh IPO config
    ipo_controlled = {
        'MA20':        _INDICATOR_COLUMNS['MA20'],
        'MA50':        _INDICATOR_COLUMNS['MA50'],
        'MA200':       _INDICATOR_COLUMNS['MA200'],
        'ADX':         _INDICATOR_COLUMNS['ADX'],
        'RSI':         _INDICATOR_COLUMNS['RSI'],
        'MACD':        _INDICATOR_COLUMNS['MACD'],
        'BB':          _INDICATOR_COLUMNS['BB'],
        'OBV':         _INDICATOR_COLUMNS['OBV'],
        'MFI':         _INDICATOR_COLUMNS['MFI'],
        'Candlestick': _INDICATOR_COLUMNS['Candlestick'],
    }

    for ipo_key, columns in ipo_controlled.items():
        if not ipo_config.get(ipo_key, False):
            _set_nan_columns(df, columns)

    # Indikator yang tidak ada di IPO config — nonaktifkan jika bukan STANDARD/FULL
    if not standard_active:
        for indicator in _ALWAYS_ON_FOR_STANDARD:
            _set_nan_columns(df, _INDICATOR_COLUMNS[indicator])


# ---------------------------------------------------------------------------
# 8.9 Tangani NaN: log indikator yang bernilai NaN di baris terakhir
# ---------------------------------------------------------------------------

def _log_nan_indicators(df: pd.DataFrame, data_age: DataAgeClassification) -> None:
    """
    Periksa baris terakhir df untuk setiap indikator.
    Log indikator yang bernilai NaN agar dapat ditampilkan sebagai st.info.
    """
    if df.empty:
        return

    last_row = df.iloc[-1]
    nan_indicators: list[str] = []

    for indicator, columns in _INDICATOR_COLUMNS.items():
        for col in columns:
            if col in df.columns and pd.isna(last_row.get(col, np.nan)):
                nan_indicators.append(indicator)
                break

    if nan_indicators:
        logger.info(
            f"[{data_age.value}] Indikator tidak tersedia (NaN): "
            + ", ".join(nan_indicators)
        )


def get_nan_indicator_messages(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> list[str]:
    """
    Kembalikan daftar pesan informatif untuk indikator yang bernilai NaN
    atau dinonaktifkan oleh IPO config.

    Digunakan oleh UI (app.py) untuk menampilkan st.info per indikator
    yang tidak tersedia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame setelah calculate_all_indicators() dipanggil.
    data_age : DataAgeClassification
        Klasifikasi usia data.

    Returns
    -------
    list[str]
        Daftar pesan, misalnya:
        ['Indikator MA200 tidak tersedia karena data tidak mencukupi']
    """
    # Nama tampilan yang lebih ramah pengguna
    _DISPLAY_NAMES: dict[str, str] = {
        'MA20':        'MA20 (Moving Average 20)',
        'MA50':        'MA50 (Moving Average 50)',
        'MA200':       'MA200 (Moving Average 200)',
        'ADX':         'ADX (Average Directional Index)',
        'RSI':         'RSI (Relative Strength Index)',
        'MACD':        'MACD',
        'ROC':         'ROC (Rate of Change)',
        'WilliamsR':   'Williams %R',
        'CCI':         'CCI (Commodity Channel Index)',
        'BB':          'Bollinger Bands',
        'ATR':         'ATR (Average True Range)',
        'OBV':         'OBV (On-Balance Volume)',
        'MFI':         'MFI (Money Flow Index)',
        'Stochastic':  'Stochastic Oscillator',
        'Candlestick': 'Pola Candlestick',
    }

    if df.empty:
        return ['Tidak ada data untuk menghitung indikator.']

    ipo_config = get_ipo_config(data_age)
    standard_active = data_age in (DataAgeClassification.STANDARD, DataAgeClassification.FULL)
    last_row = df.iloc[-1]
    messages: list[str] = []

    for indicator, columns in _INDICATOR_COLUMNS.items():
        display_name = _DISPLAY_NAMES.get(indicator, indicator)

        # Cek apakah dinonaktifkan oleh IPO config
        ipo_key = _get_ipo_key(indicator)
        if ipo_key is not None and not ipo_config.get(ipo_key, False):
            messages.append(
                f'Indikator {display_name} tidak tersedia karena data tidak mencukupi '
                f'(klasifikasi: {data_age.value})'
            )
            continue

        if indicator in _ALWAYS_ON_FOR_STANDARD and not standard_active:
            messages.append(
                f'Indikator {display_name} tidak tersedia karena data tidak mencukupi '
                f'(klasifikasi: {data_age.value})'
            )
            continue

        # Cek apakah kolom ada dan bernilai NaN di baris terakhir
        for col in columns:
            # Cek nama kolom eksak dulu, lalu cari dengan prefix
            actual_col = col
            if col not in df.columns:
                # Cari berdasarkan prefix (misal 'BBL' dari 'BBL_20_2.0')
                prefix = col.split('_')[0]
                found = next((c for c in df.columns if c.startswith(prefix + '_')), None)
                if found:
                    actual_col = found
                else:
                    messages.append(
                        f'Indikator {display_name} tidak tersedia karena data tidak mencukupi'
                    )
                    break

            if pd.isna(last_row.get(actual_col, np.nan)):
                messages.append(
                    f'Indikator {display_name} tidak tersedia karena data tidak mencukupi'
                )
                break

    return messages
