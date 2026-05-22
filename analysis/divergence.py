"""
analysis/divergence.py — Deteksi Divergensi Harga-Indikator

Mendeteksi empat jenis divergensi:
  - Bullish Regular  : harga LL, indikator HL  → potensi pembalikan naik
  - Bearish Regular  : harga HH, indikator LH  → potensi pembalikan turun
  - Hidden Bullish   : harga HL, indikator LL  → konfirmasi tren naik
  - Hidden Bearish   : harga LH, indikator HH  → konfirmasi tren turun

Indikator yang digunakan: RSI, MACD histogram, MFI.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from models import DivergenceResult, DivergenceType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

_PIVOT_WINDOW = 5          # Jumlah candle kiri/kanan untuk menentukan pivot
_MIN_PIVOT_DISTANCE = 5    # Jarak minimum antar pivot (candle)
_MIN_SLOPE_DIFF = 0.001    # Perbedaan slope minimum agar dianggap divergensi


# ---------------------------------------------------------------------------
# Fungsi publik utama
# ---------------------------------------------------------------------------

def detect_divergences(df: pd.DataFrame) -> list[DivergenceResult]:
    """
    Deteksi semua divergensi pada DataFrame OHLCV yang sudah berisi kolom indikator.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom: Close, RSI_14, MACDh_12_26_9, MFI_14.

    Returns
    -------
    list[DivergenceResult]
        Daftar divergensi yang terdeteksi. Kosong jika tidak ada.
    """
    if df is None or df.empty or len(df) < (_PIVOT_WINDOW * 2 + _MIN_PIVOT_DISTANCE + 1):
        return []

    results: list[DivergenceResult] = []

    # Pasangan (nama_indikator, kolom_df)
    indicator_pairs = [
        ('RSI',            'RSI_14'),
        ('MACD Histogram', 'MACDh_12_26_9'),
        ('MFI',            'MFI_14'),
    ]

    price_series = df['Close'].copy()

    for indicator_name, col in indicator_pairs:
        if col not in df.columns:
            continue
        indicator_series = df[col].copy()

        # Lewati jika terlalu banyak NaN
        if indicator_series.isna().sum() > len(indicator_series) * 0.5:
            continue

        # Isi NaN dengan forward-fill agar pivot tidak terganggu
        indicator_series = indicator_series.ffill().bfill()

        divergences = _detect_divergences_for_indicator(
            price_series=price_series,
            indicator_series=indicator_series,
            indicator_name=indicator_name,
        )
        results.extend(divergences)

    return results


# ---------------------------------------------------------------------------
# Fungsi privat: deteksi per indikator
# ---------------------------------------------------------------------------

def _detect_divergences_for_indicator(
    price_series: pd.Series,
    indicator_series: pd.Series,
    indicator_name: str,
) -> list[DivergenceResult]:
    """Deteksi semua jenis divergensi untuk satu indikator."""
    results: list[DivergenceResult] = []

    # Cari pivot high dan low
    price_highs = _find_pivot_highs(price_series, window=_PIVOT_WINDOW)
    price_lows  = _find_pivot_lows(price_series,  window=_PIVOT_WINDOW)
    ind_highs   = _find_pivot_highs(indicator_series, window=_PIVOT_WINDOW)
    ind_lows    = _find_pivot_lows(indicator_series,  window=_PIVOT_WINDOW)

    # Butuh minimal 2 pivot untuk membandingkan slope
    if len(price_highs) >= 2 and len(ind_highs) >= 2:
        # Bearish Regular: harga HH, indikator LH
        div = _check_bearish_regular(price_highs, ind_highs, price_series, indicator_series, indicator_name)
        if div:
            results.append(div)

        # Hidden Bearish: harga LH, indikator HH
        div = _check_hidden_bearish(price_highs, ind_highs, price_series, indicator_series, indicator_name)
        if div:
            results.append(div)

    if len(price_lows) >= 2 and len(ind_lows) >= 2:
        # Bullish Regular: harga LL, indikator HL
        div = _check_bullish_regular(price_lows, ind_lows, price_series, indicator_series, indicator_name)
        if div:
            results.append(div)

        # Hidden Bullish: harga HL, indikator LL
        div = _check_hidden_bullish(price_lows, ind_lows, price_series, indicator_series, indicator_name)
        if div:
            results.append(div)

    return results


# ---------------------------------------------------------------------------
# Task 9.1 — Pivot High dan Pivot Low
# ---------------------------------------------------------------------------

def _find_pivot_highs(series: pd.Series, window: int = 5) -> list[tuple[int, float]]:
    """
    Cari local maxima pada series.

    Sebuah titik i dianggap pivot high jika:
        series[i] == max(series[i-window : i+window+1])

    Parameters
    ----------
    series : pd.Series
        Series numerik (harga atau indikator).
    window : int
        Jumlah candle kiri dan kanan yang dibandingkan.

    Returns
    -------
    list[tuple[int, float]]
        Daftar (indeks_integer, nilai) pivot high, diurutkan dari terlama ke terbaru.
    """
    pivots: list[tuple[int, float]] = []
    values = series.values
    n = len(values)

    for i in range(window, n - window):
        if np.isnan(values[i]):
            continue
        window_slice = values[max(0, i - window): i + window + 1]
        # Abaikan NaN dalam window
        valid = window_slice[~np.isnan(window_slice)]
        if len(valid) == 0:
            continue
        if values[i] == np.max(valid) and np.sum(window_slice == values[i]) == 1:
            pivots.append((i, float(values[i])))

    return pivots


def _find_pivot_lows(series: pd.Series, window: int = 5) -> list[tuple[int, float]]:
    """
    Cari local minima pada series.

    Sebuah titik i dianggap pivot low jika:
        series[i] == min(series[i-window : i+window+1])

    Parameters
    ----------
    series : pd.Series
        Series numerik (harga atau indikator).
    window : int
        Jumlah candle kiri dan kanan yang dibandingkan.

    Returns
    -------
    list[tuple[int, float]]
        Daftar (indeks_integer, nilai) pivot low, diurutkan dari terlama ke terbaru.
    """
    pivots: list[tuple[int, float]] = []
    values = series.values
    n = len(values)

    for i in range(window, n - window):
        if np.isnan(values[i]):
            continue
        window_slice = values[max(0, i - window): i + window + 1]
        valid = window_slice[~np.isnan(window_slice)]
        if len(valid) == 0:
            continue
        if values[i] == np.min(valid) and np.sum(window_slice == values[i]) == 1:
            pivots.append((i, float(values[i])))

    return pivots


# ---------------------------------------------------------------------------
# Helper: ambil 2 pivot terakhir yang cukup berjauhan
# ---------------------------------------------------------------------------

def _get_last_two_pivots(
    pivots: list[tuple[int, float]],
    min_distance: int = _MIN_PIVOT_DISTANCE,
) -> Optional[tuple[tuple[int, float], tuple[int, float]]]:
    """
    Ambil dua pivot terakhir yang berjarak minimal min_distance candle.

    Returns
    -------
    tuple (pivot_lama, pivot_baru) atau None jika tidak cukup pivot.
    """
    if len(pivots) < 2:
        return None

    # Pivot terbaru adalah yang terakhir
    pivot_baru = pivots[-1]

    # Cari pivot sebelumnya yang cukup jauh
    for i in range(len(pivots) - 2, -1, -1):
        pivot_lama = pivots[i]
        if pivot_baru[0] - pivot_lama[0] >= min_distance:
            return (pivot_lama, pivot_baru)

    return None


# ---------------------------------------------------------------------------
# Task 9.7 — Hitung divergence strength
# ---------------------------------------------------------------------------

def _calculate_strength(
    price_pivot_old: float,
    price_pivot_new: float,
    ind_pivot_old: float,
    ind_pivot_new: float,
    price_idx_old: int,
    price_idx_new: int,
) -> float:
    """
    Hitung kekuatan divergensi dari perbedaan slope harga dan indikator.

    strength = abs(slope_price - slope_indicator) / normalization_factor

    Returns
    -------
    float
        Nilai kekuatan divergensi dalam range [0, 1].
    """
    distance = max(price_idx_new - price_idx_old, 1)

    # Normalisasi slope ke persentase perubahan
    slope_price = (price_pivot_new - price_pivot_old) / (abs(price_pivot_old) * distance + 1e-10)
    slope_ind   = (ind_pivot_new - ind_pivot_old) / (abs(ind_pivot_old) * distance + 1e-10)

    raw_strength = abs(slope_price - slope_ind)

    # Clamp ke [0, 1]
    return float(min(raw_strength * 10, 1.0))


# ---------------------------------------------------------------------------
# Task 9.2 — Bullish Regular Divergence
# ---------------------------------------------------------------------------

def _check_bullish_regular(
    price_lows: list[tuple[int, float]],
    ind_lows: list[tuple[int, float]],
    price_series: pd.Series,
    indicator_series: pd.Series,
    indicator_name: str,
) -> Optional[DivergenceResult]:
    """
    Bullish Regular: harga membentuk Lower Low, indikator membentuk Higher Low.
    Sinyal: potensi pembalikan naik.
    """
    price_pair = _get_last_two_pivots(price_lows)
    ind_pair   = _get_last_two_pivots(ind_lows)

    if price_pair is None or ind_pair is None:
        return None

    (p_idx_old, p_val_old), (p_idx_new, p_val_new) = price_pair
    (i_idx_old, i_val_old), (i_idx_new, i_val_new) = ind_pair

    # Harga: Lower Low (turun)
    price_slope = p_val_new - p_val_old
    # Indikator: Higher Low (naik)
    ind_slope = i_val_new - i_val_old

    if price_slope >= 0 or ind_slope <= 0:
        return None

    if abs(price_slope) < _MIN_SLOPE_DIFF * abs(p_val_old + 1e-10):
        return None

    strength = _calculate_strength(p_val_old, p_val_new, i_val_old, i_val_new, p_idx_old, p_idx_new)

    return DivergenceResult(
        divergence_type=DivergenceType.BULLISH_REGULAR,
        strength=strength,
        indicators_confirming=[indicator_name],
        price_pivot_indices=[p_idx_old, p_idx_new],
        indicator_pivot_indices=[i_idx_old, i_idx_new],
        price_pivot_values=[p_val_old, p_val_new],
        indicator_pivot_values=[i_val_old, i_val_new],
    )


# ---------------------------------------------------------------------------
# Task 9.3 — Bearish Regular Divergence
# ---------------------------------------------------------------------------

def _check_bearish_regular(
    price_highs: list[tuple[int, float]],
    ind_highs: list[tuple[int, float]],
    price_series: pd.Series,
    indicator_series: pd.Series,
    indicator_name: str,
) -> Optional[DivergenceResult]:
    """
    Bearish Regular: harga membentuk Higher High, indikator membentuk Lower High.
    Sinyal: potensi pembalikan turun.
    """
    price_pair = _get_last_two_pivots(price_highs)
    ind_pair   = _get_last_two_pivots(ind_highs)

    if price_pair is None or ind_pair is None:
        return None

    (p_idx_old, p_val_old), (p_idx_new, p_val_new) = price_pair
    (i_idx_old, i_val_old), (i_idx_new, i_val_new) = ind_pair

    # Harga: Higher High (naik)
    price_slope = p_val_new - p_val_old
    # Indikator: Lower High (turun)
    ind_slope = i_val_new - i_val_old

    if price_slope <= 0 or ind_slope >= 0:
        return None

    if abs(price_slope) < _MIN_SLOPE_DIFF * abs(p_val_old + 1e-10):
        return None

    strength = _calculate_strength(p_val_old, p_val_new, i_val_old, i_val_new, p_idx_old, p_idx_new)

    return DivergenceResult(
        divergence_type=DivergenceType.BEARISH_REGULAR,
        strength=strength,
        indicators_confirming=[indicator_name],
        price_pivot_indices=[p_idx_old, p_idx_new],
        indicator_pivot_indices=[i_idx_old, i_idx_new],
        price_pivot_values=[p_val_old, p_val_new],
        indicator_pivot_values=[i_val_old, i_val_new],
    )


# ---------------------------------------------------------------------------
# Task 9.4 — Hidden Bullish Divergence
# ---------------------------------------------------------------------------

def _check_hidden_bullish(
    price_lows: list[tuple[int, float]],
    ind_lows: list[tuple[int, float]],
    price_series: pd.Series,
    indicator_series: pd.Series,
    indicator_name: str,
) -> Optional[DivergenceResult]:
    """
    Hidden Bullish: harga membentuk Higher Low (dalam uptrend),
    indikator membentuk Lower Low.
    Sinyal: konfirmasi kelanjutan tren naik.
    """
    price_pair = _get_last_two_pivots(price_lows)
    ind_pair   = _get_last_two_pivots(ind_lows)

    if price_pair is None or ind_pair is None:
        return None

    (p_idx_old, p_val_old), (p_idx_new, p_val_new) = price_pair
    (i_idx_old, i_val_old), (i_idx_new, i_val_new) = ind_pair

    # Harga: Higher Low (naik)
    price_slope = p_val_new - p_val_old
    # Indikator: Lower Low (turun)
    ind_slope = i_val_new - i_val_old

    if price_slope <= 0 or ind_slope >= 0:
        return None

    if abs(price_slope) < _MIN_SLOPE_DIFF * abs(p_val_old + 1e-10):
        return None

    strength = _calculate_strength(p_val_old, p_val_new, i_val_old, i_val_new, p_idx_old, p_idx_new)

    return DivergenceResult(
        divergence_type=DivergenceType.BULLISH_HIDDEN,
        strength=strength,
        indicators_confirming=[indicator_name],
        price_pivot_indices=[p_idx_old, p_idx_new],
        indicator_pivot_indices=[i_idx_old, i_idx_new],
        price_pivot_values=[p_val_old, p_val_new],
        indicator_pivot_values=[i_val_old, i_val_new],
    )


# ---------------------------------------------------------------------------
# Task 9.5 — Hidden Bearish Divergence
# ---------------------------------------------------------------------------

def _check_hidden_bearish(
    price_highs: list[tuple[int, float]],
    ind_highs: list[tuple[int, float]],
    price_series: pd.Series,
    indicator_series: pd.Series,
    indicator_name: str,
) -> Optional[DivergenceResult]:
    """
    Hidden Bearish: harga membentuk Lower High (dalam downtrend),
    indikator membentuk Higher High.
    Sinyal: konfirmasi kelanjutan tren turun.
    """
    price_pair = _get_last_two_pivots(price_highs)
    ind_pair   = _get_last_two_pivots(ind_highs)

    if price_pair is None or ind_pair is None:
        return None

    (p_idx_old, p_val_old), (p_idx_new, p_val_new) = price_pair
    (i_idx_old, i_val_old), (i_idx_new, i_val_new) = ind_pair

    # Harga: Lower High (turun)
    price_slope = p_val_new - p_val_old
    # Indikator: Higher High (naik)
    ind_slope = i_val_new - i_val_old

    if price_slope >= 0 or ind_slope <= 0:
        return None

    if abs(price_slope) < _MIN_SLOPE_DIFF * abs(p_val_old + 1e-10):
        return None

    strength = _calculate_strength(p_val_old, p_val_new, i_val_old, i_val_new, p_idx_old, p_idx_new)

    return DivergenceResult(
        divergence_type=DivergenceType.BEARISH_HIDDEN,
        strength=strength,
        indicators_confirming=[indicator_name],
        price_pivot_indices=[p_idx_old, p_idx_new],
        indicator_pivot_indices=[i_idx_old, i_idx_new],
        price_pivot_values=[p_val_old, p_val_new],
        indicator_pivot_values=[i_val_old, i_val_new],
    )
