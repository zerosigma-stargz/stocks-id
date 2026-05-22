"""
analysis/support_resistance.py — Level Support & Resistance

Menghitung level S/R dari empat sumber:
  1. Statis   : swing high/low historis dengan minimal 2 sentuhan
  2. Dinamis  : nilai MA20, MA50, MA200 pada candle terakhir
  3. Fibonacci: retracement 23.6%, 38.2%, 50%, 61.8%, 78.6%
  4. Psikologis: kelipatan 100, 500, 1000 dalam rentang harga ±20%
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logging.warning("scipy tidak tersedia. Level S/R statis tidak dapat dihitung.")

from models import (
    DataAgeClassification,
    SRLevelSource,
    SRLevelType,
    SRResult,
    SupportResistanceLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

_FIBONACCI_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]
_SWING_LOOKBACK   = 100   # Candle terakhir untuk menentukan swing high/low Fibonacci
_STATIC_ORDER     = 5     # Window argrelextrema untuk level statis
_TOUCH_TOLERANCE  = 0.005 # 0.5% toleransi untuk menghitung sentuhan


# ---------------------------------------------------------------------------
# Fungsi publik utama
# ---------------------------------------------------------------------------

def calculate_sr_levels(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> SRResult:
    """
    Hitung semua level Support & Resistance dan kembalikan SRResult.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV dengan kolom indikator (SMA_20, SMA_50, SMA_200,
        ATRr_14).
    data_age : DataAgeClassification
        Klasifikasi usia data untuk menentukan indikator yang tersedia.

    Returns
    -------
    SRResult
        Semua level, level support terdekat, level resistance terdekat,
        harga saat ini, dan nilai ATR.
    """
    if df is None or df.empty:
        return _empty_sr_result(0.0, 0.0)

    current_price = float(df['Close'].iloc[-1])
    atr = _get_atr(df)

    all_levels: list[SupportResistanceLevel] = []

    # 1. Level statis dari swing high/low
    all_levels.extend(_find_static_levels(df, current_price, atr))

    # 2. Level dinamis dari MA
    all_levels.extend(_get_dynamic_levels(df, current_price, atr))

    # 3. Level Fibonacci
    if data_age in (DataAgeClassification.STANDARD, DataAgeClassification.FULL):
        all_levels.extend(_calculate_fibonacci_levels(df, current_price, atr))

    # 4. Level psikologis
    all_levels.extend(_get_psychological_levels(current_price, atr))

    # Hapus duplikat level yang terlalu berdekatan (dalam 0.5%)
    all_levels = _deduplicate_levels(all_levels)

    nearest_support    = _find_nearest_support(all_levels, current_price)
    nearest_resistance = _find_nearest_resistance(all_levels, current_price)

    return SRResult(
        all_levels=all_levels,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        current_price=current_price,
        atr=atr,
    )


# ---------------------------------------------------------------------------
# Task 10.1 — Level Statis (swing high/low)
# ---------------------------------------------------------------------------

def _find_static_levels(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    min_touches: int = 2,
) -> list[SupportResistanceLevel]:
    """
    Cari swing high dan swing low historis yang disentuh minimal min_touches kali.

    Menggunakan scipy.signal.argrelextrema untuk menemukan extrema lokal.
    """
    levels: list[SupportResistanceLevel] = []

    if not _SCIPY_AVAILABLE or df.empty:
        return levels

    high_arr  = df['High'].values
    low_arr   = df['Low'].values
    close_arr = df['Close'].values
    n = len(df)

    if n < _STATIC_ORDER * 2 + 1:
        return levels

    # Cari indeks swing high dan swing low
    swing_high_idx = argrelextrema(high_arr,  np.greater_equal, order=_STATIC_ORDER)[0]
    swing_low_idx  = argrelextrema(low_arr,   np.less_equal,    order=_STATIC_ORDER)[0]

    tolerance = current_price * _TOUCH_TOLERANCE

    # Proses swing high → Resistance
    for idx in swing_high_idx:
        price_level = float(high_arr[idx])
        touches = _count_touches(close_arr, price_level, tolerance)
        if touches >= min_touches:
            level_type = (
                SRLevelType.RESISTANCE if price_level > current_price
                else SRLevelType.SUPPORT
            )
            levels.append(_make_level(
                price=price_level,
                level_type=level_type,
                source=SRLevelSource.STATIC,
                source_detail=f'Swing High (idx={idx})',
                touches=touches,
                current_price=current_price,
                atr=atr,
            ))

    # Proses swing low → Support
    for idx in swing_low_idx:
        price_level = float(low_arr[idx])
        touches = _count_touches(close_arr, price_level, tolerance)
        if touches >= min_touches:
            level_type = (
                SRLevelType.SUPPORT if price_level < current_price
                else SRLevelType.RESISTANCE
            )
            levels.append(_make_level(
                price=price_level,
                level_type=level_type,
                source=SRLevelSource.STATIC,
                source_detail=f'Swing Low (idx={idx})',
                touches=touches,
                current_price=current_price,
                atr=atr,
            ))

    return levels


# ---------------------------------------------------------------------------
# Task 10.2 — Level Dinamis (MA)
# ---------------------------------------------------------------------------

def _get_dynamic_levels(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
) -> list[SupportResistanceLevel]:
    """Ambil nilai SMA_20, SMA_50, SMA_200 dari candle terakhir sebagai level dinamis."""
    levels: list[SupportResistanceLevel] = []
    last = df.iloc[-1]

    ma_map = {
        'SMA_20':  'MA20',
        'SMA_50':  'MA50',
        'SMA_200': 'MA200',
    }

    for col, label in ma_map.items():
        if col not in df.columns:
            continue
        val = last.get(col, np.nan)
        if pd.isna(val):
            continue
        price_level = float(val)
        level_type = (
            SRLevelType.SUPPORT if price_level < current_price
            else SRLevelType.RESISTANCE
        )
        levels.append(_make_level(
            price=price_level,
            level_type=level_type,
            source=SRLevelSource.DYNAMIC_MA,
            source_detail=label,
            touches=1,
            current_price=current_price,
            atr=atr,
        ))

    return levels


# ---------------------------------------------------------------------------
# Task 10.3 — Level Fibonacci
# ---------------------------------------------------------------------------

def _calculate_fibonacci_levels(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
) -> list[SupportResistanceLevel]:
    """
    Hitung level Fibonacci Retracement dari swing high dan swing low
    dalam _SWING_LOOKBACK candle terakhir.
    """
    levels: list[SupportResistanceLevel] = []

    lookback = df.tail(_SWING_LOOKBACK)
    if lookback.empty:
        return levels

    swing_high = float(lookback['High'].max())
    swing_low  = float(lookback['Low'].min())
    price_range = swing_high - swing_low

    if price_range <= 0:
        return levels

    for ratio in _FIBONACCI_RATIOS:
        fib_price = swing_low + price_range * ratio
        level_type = (
            SRLevelType.SUPPORT if fib_price < current_price
            else SRLevelType.RESISTANCE
        )
        levels.append(_make_level(
            price=fib_price,
            level_type=level_type,
            source=SRLevelSource.FIBONACCI,
            source_detail=f'Fib {ratio*100:.1f}%',
            touches=1,
            current_price=current_price,
            atr=atr,
        ))

    return levels


# ---------------------------------------------------------------------------
# Task 10.4 — Level Psikologis
# ---------------------------------------------------------------------------

def _get_psychological_levels(
    current_price: float,
    atr: float,
) -> list[SupportResistanceLevel]:
    """
    Generate level psikologis: kelipatan 100, 500, 1000 dalam rentang ±20% harga.
    """
    levels: list[SupportResistanceLevel] = []
    price_min = current_price * 0.80
    price_max = current_price * 1.20

    # Tentukan kelipatan berdasarkan skala harga
    if current_price >= 10_000:
        multiples = [1000, 5000, 10_000]
    elif current_price >= 1_000:
        multiples = [100, 500, 1000]
    elif current_price >= 100:
        multiples = [50, 100, 500]
    else:
        multiples = [10, 25, 50]

    seen: set[float] = set()

    for multiple in multiples:
        start = int(price_min / multiple) * multiple
        end   = int(price_max / multiple) * multiple + multiple

        for level_price in range(start, end + multiple, multiple):
            p = float(level_price)
            if p <= 0 or p in seen:
                continue
            if not (price_min <= p <= price_max):
                continue
            seen.add(p)
            level_type = (
                SRLevelType.SUPPORT if p < current_price
                else SRLevelType.RESISTANCE
            )
            levels.append(_make_level(
                price=p,
                level_type=level_type,
                source=SRLevelSource.PSYCHOLOGICAL,
                source_detail=f'Psikologis {multiple}',
                touches=1,
                current_price=current_price,
                atr=atr,
            ))

    return levels


# ---------------------------------------------------------------------------
# Task 10.6 — Nearest Support & Resistance
# ---------------------------------------------------------------------------

def _find_nearest_support(
    levels: list[SupportResistanceLevel],
    current_price: float,
) -> Optional[SupportResistanceLevel]:
    """
    Cari level support terdekat di bawah harga saat ini.
    Kembalikan level dengan harga tertinggi di antara semua support.
    """
    supports = [
        lvl for lvl in levels
        if lvl.level_type == SRLevelType.SUPPORT and lvl.price < current_price
    ]
    if not supports:
        return None
    return max(supports, key=lambda x: x.price)


def _find_nearest_resistance(
    levels: list[SupportResistanceLevel],
    current_price: float,
) -> Optional[SupportResistanceLevel]:
    """
    Cari level resistance terdekat di atas harga saat ini.
    Kembalikan level dengan harga terendah di antara semua resistance.
    """
    resistances = [
        lvl for lvl in levels
        if lvl.level_type == SRLevelType.RESISTANCE and lvl.price > current_price
    ]
    if not resistances:
        return None
    return min(resistances, key=lambda x: x.price)


# ---------------------------------------------------------------------------
# Task 10.7 — Hitung distance_pct dan distance_atr
# ---------------------------------------------------------------------------

def _make_level(
    price: float,
    level_type: SRLevelType,
    source: SRLevelSource,
    source_detail: str,
    touches: int,
    current_price: float,
    atr: float,
) -> SupportResistanceLevel:
    """Buat SupportResistanceLevel dengan distance_pct dan distance_atr."""
    if current_price > 0:
        distance_pct = abs(price - current_price) / current_price * 100
    else:
        distance_pct = 0.0

    if atr > 0:
        distance_atr = abs(price - current_price) / atr
    else:
        distance_atr = 0.0

    return SupportResistanceLevel(
        price=price,
        level_type=level_type,
        source=source,
        source_detail=source_detail,
        touches=touches,
        distance_pct=round(distance_pct, 2),
        distance_atr=round(distance_atr, 2),
    )


# ---------------------------------------------------------------------------
# Helper privat
# ---------------------------------------------------------------------------

def _get_atr(df: pd.DataFrame) -> float:
    """Ambil nilai ATR terakhir dari kolom ATRr_14, atau 0 jika tidak tersedia."""
    if 'ATRr_14' not in df.columns:
        return 0.0
    val = df['ATRr_14'].iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def _count_touches(
    close_arr: np.ndarray,
    price_level: float,
    tolerance: float,
) -> int:
    """Hitung berapa kali harga close menyentuh level dalam toleransi."""
    return int(np.sum(np.abs(close_arr - price_level) <= tolerance))


def _deduplicate_levels(
    levels: list[SupportResistanceLevel],
    tolerance_pct: float = 0.005,
) -> list[SupportResistanceLevel]:
    """
    Hapus level duplikat yang harganya terlalu berdekatan (dalam tolerance_pct).
    Pertahankan level dengan touches terbanyak.
    """
    if not levels:
        return levels

    # Urutkan berdasarkan harga
    sorted_levels = sorted(levels, key=lambda x: x.price)
    result: list[SupportResistanceLevel] = [sorted_levels[0]]

    for lvl in sorted_levels[1:]:
        last = result[-1]
        if last.price > 0:
            diff_pct = abs(lvl.price - last.price) / last.price
        else:
            diff_pct = 1.0

        if diff_pct <= tolerance_pct:
            # Pertahankan yang memiliki lebih banyak sentuhan
            if lvl.touches > last.touches:
                result[-1] = lvl
        else:
            result.append(lvl)

    return result


def _empty_sr_result(current_price: float, atr: float) -> SRResult:
    """Kembalikan SRResult kosong jika data tidak tersedia."""
    return SRResult(
        all_levels=[],
        nearest_support=None,
        nearest_resistance=None,
        current_price=current_price,
        atr=atr,
    )
