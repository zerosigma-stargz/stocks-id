"""
analysis/signals.py — Mesin Sinyal 5 Dimensi

Menghitung skor sinyal dari 5 dimensi untuk setiap timeframe:
  Dimensi 1 — Tren        (max 5 poin)
  Dimensi 2 — Momentum    (max 10 poin)
  Dimensi 3 — Volatilitas (max 6 poin)
  Dimensi 4 — Volume      (max 7 poin)
  Dimensi 5 — Candlestick (max 3 poin)

Total maksimum: 31 poin.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from models import (
    DataAgeClassification,
    DimensionScore,
    DivergenceResult,
    DivergenceType,
    MarketRegime,
    MarketRegimeResult,
    SignalDirection,
    SignalResult,
    SignalStrength,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta bobot dimensi
# ---------------------------------------------------------------------------

_DIM_WEIGHTS = {1: 0.25, 2: 0.25, 3: 0.20, 4: 0.20, 5: 0.10}
_DIM_MAX     = {1: 5.0,  2: 10.0, 3: 6.0,  4: 7.0,  5: 3.0}
_TOTAL_MAX   = 31.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_val(row: pd.Series, col: str) -> float:
    """Ambil nilai float dari baris, kembalikan nan jika tidak ada atau NaN."""
    val = row.get(col, np.nan)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float('nan')
    try:
        return float(val)
    except (TypeError, ValueError):
        return float('nan')


def _get_val_flexible(row: pd.Series, candidates: list[str]) -> float:
    """
    Ambil nilai float dari baris menggunakan daftar nama kolom kandidat.
    Kembalikan nilai pertama yang tidak NaN, atau nan jika semua NaN.
    """
    for col in candidates:
        val = _get_val(row, col)
        if not np.isnan(val):
            return val
    return float('nan')


def _neutral_signal(timeframe: str) -> SignalResult:
    """Kembalikan SignalResult netral jika data tidak tersedia."""
    dims = [
        DimensionScore(d, name, 0.0, mx, SignalDirection.NETRAL, [], w)
        for d, name, mx, w in [
            (1, 'Tren',        5.0,  0.25),
            (2, 'Momentum',    10.0, 0.25),
            (3, 'Volatilitas', 6.0,  0.20),
            (4, 'Volume',      7.0,  0.20),
            (5, 'Candlestick', 3.0,  0.10),
        ]
    ]
    return SignalResult(
        dimension_scores=dims,
        total_score=0.0,
        signal_strength=SignalStrength.NEUTRAL,
        direction=SignalDirection.NETRAL,
        timeframe=timeframe,
        alignment_score=0.0,
    )


def _classify_strength(total_score: float) -> SignalStrength:
    """Klasifikasi kekuatan sinyal berdasarkan total skor."""
    if total_score >= 25:
        return SignalStrength.STRONG
    elif total_score >= 17:
        return SignalStrength.MODERATE
    elif total_score >= 10:
        return SignalStrength.WEAK
    else:
        return SignalStrength.NEUTRAL


def _determine_direction(dimension_scores: list[DimensionScore]) -> SignalDirection:
    """Tentukan arah sinyal dari mayoritas dimensi yang memiliki skor > 0."""
    beli_count = sum(
        1 for d in dimension_scores
        if d.direction == SignalDirection.BELI and d.score > 0
    )
    jual_count = sum(
        1 for d in dimension_scores
        if d.direction == SignalDirection.JUAL and d.score > 0
    )
    if beli_count > jual_count:
        return SignalDirection.BELI
    elif jual_count > beli_count:
        return SignalDirection.JUAL
    else:
        return SignalDirection.NETRAL


def _apply_regime_filter(
    direction: SignalDirection,
    regime: MarketRegime,
    total_score: float,
) -> SignalDirection:
    """
    Task 11.7 — Terapkan filter kondisi pasar.
    BULL_TREND: paksa BELI jika skor cukup, NETRAL jika tidak.
    BEAR_TREND: paksa JUAL jika skor cukup, NETRAL jika tidak.
    """
    if regime == MarketRegime.BULL_TREND:
        if direction == SignalDirection.JUAL:
            return SignalDirection.NETRAL
    elif regime == MarketRegime.BEAR_TREND:
        if direction == SignalDirection.BELI:
            return SignalDirection.NETRAL
    return direction


# ---------------------------------------------------------------------------
# Task 11.6 — Fungsi utama: calculate_signal
# ---------------------------------------------------------------------------

def calculate_signal(
    df: pd.DataFrame,
    divergences: list[DivergenceResult],
    data_age: DataAgeClassification,
    regime: MarketRegimeResult,
    timeframe: str = 'daily',
) -> SignalResult:
    """
    Hitung sinyal 5 dimensi untuk satu timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV dengan kolom indikator yang sudah dihitung.
    divergences : list[DivergenceResult]
        Daftar divergensi yang terdeteksi.
    data_age : DataAgeClassification
        Klasifikasi usia data.
    regime : MarketRegimeResult
        Hasil deteksi kondisi pasar.
    timeframe : str
        Label timeframe ('daily', 'weekly', 'monthly').

    Returns
    -------
    SignalResult
    """
    if df is None or df.empty:
        return _neutral_signal(timeframe)

    dim1 = calculate_dimension1_trend(df, data_age)
    dim2 = calculate_dimension2_momentum(df, divergences, data_age)
    dim3 = calculate_dimension3_volatility(df, data_age)
    dim4 = calculate_dimension4_volume(df, data_age)
    dim5 = calculate_dimension5_candlestick(df, data_age)

    dimension_scores = [dim1, dim2, dim3, dim4, dim5]

    total_score = sum(d.score for d in dimension_scores)
    total_score = max(0.0, min(total_score, _TOTAL_MAX))

    signal_strength = _classify_strength(total_score)
    direction = _determine_direction(dimension_scores)
    direction = _apply_regime_filter(direction, regime.regime, total_score)

    return SignalResult(
        dimension_scores=dimension_scores,
        total_score=round(total_score, 2),
        signal_strength=signal_strength,
        direction=direction,
        timeframe=timeframe,
        alignment_score=0.0,
    )


# ---------------------------------------------------------------------------
# Task 11.1 — Dimensi 1: Tren (max 7 poin)
# ---------------------------------------------------------------------------

def calculate_dimension1_trend(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> DimensionScore:
    """
    Dimensi 1 — Tren (max 5 poin):
      +3: Triple MA alignment (MA20>MA50>MA200 bullish, atau sebaliknya)
      +2: ADX > 25 dengan +DI > -DI (bull) atau -DI > +DI (bear)
    """
    score = 0.0
    direction = SignalDirection.NETRAL
    triggered: list[str] = []
    last = df.iloc[-1]
    close_price = float(df['Close'].iloc[-1])

    # --- Triple MA alignment (+3) ---
    sma20  = _get_val(last, 'SMA_20')
    sma50  = _get_val(last, 'SMA_50')
    sma200 = _get_val(last, 'SMA_200')

    if not any(np.isnan(v) for v in [sma20, sma50, sma200]):
        if sma20 > sma50 > sma200:
            score += 3.0
            direction = SignalDirection.BELI
            triggered.append('Triple MA Bullish (MA20>MA50>MA200)')
        elif sma20 < sma50 < sma200:
            score += 3.0
            direction = SignalDirection.JUAL
            triggered.append('Triple MA Bearish (MA20<MA50<MA200)')
    elif not np.isnan(sma20) and not np.isnan(sma50):
        if sma20 > sma50 and close_price > sma20:
            score += 1.5
            direction = SignalDirection.BELI
            triggered.append('MA20>MA50 Bullish')
        elif sma20 < sma50 and close_price < sma20:
            score += 1.5
            direction = SignalDirection.JUAL
            triggered.append('MA20<MA50 Bearish')

    # --- ADX + DI (+2) ---
    adx = _get_val(last, 'ADX_14')
    dmp = _get_val(last, 'DMP_14')
    dmn = _get_val(last, 'DMN_14')

    if not any(np.isnan(v) for v in [adx, dmp, dmn]) and adx > 25:
        if dmp > dmn:
            score += 2.0
            if direction == SignalDirection.NETRAL:
                direction = SignalDirection.BELI
            triggered.append(f'ADX Bullish (ADX={adx:.1f}, +DI={dmp:.1f} > -DI={dmn:.1f})')
        elif dmn > dmp:
            score += 2.0
            if direction == SignalDirection.NETRAL:
                direction = SignalDirection.JUAL
            triggered.append(f'ADX Bearish (ADX={adx:.1f}, -DI={dmn:.1f} > +DI={dmp:.1f})')

    score = min(score, _DIM_MAX[1])

    return DimensionScore(
        dimension=1,
        name='Tren',
        score=round(score, 2),
        max_score=_DIM_MAX[1],
        direction=direction,
        triggered_indicators=triggered,
        weight=_DIM_WEIGHTS[1],
    )


# ---------------------------------------------------------------------------
# Task 11.2 — Dimensi 2: Momentum (max 10 poin)
# ---------------------------------------------------------------------------

def calculate_dimension2_momentum(
    df: pd.DataFrame,
    divergences: list[DivergenceResult],
    data_age: DataAgeClassification,
) -> DimensionScore:
    """
    Dimensi 2 — Momentum (max 10 poin):
      +2: RSI < 30 (oversold=bull) atau RSI > 70 (overbought=bear)
      +3: RSI divergence terdeteksi
      +2: MACD crossover (golden/death cross)
      +3: MACD zero-line cross + histogram momentum searah
    """
    score = 0.0
    direction = SignalDirection.NETRAL
    triggered: list[str] = []
    last = df.iloc[-1]

    # --- RSI extreme (+2) ---
    rsi = _get_val(last, 'RSI_14')
    if not np.isnan(rsi):
        if rsi < 30:
            score += 2.0
            direction = SignalDirection.BELI
            triggered.append(f'RSI Oversold ({rsi:.1f} < 30)')
        elif rsi > 70:
            score += 2.0
            direction = SignalDirection.JUAL
            triggered.append(f'RSI Overbought ({rsi:.1f} > 70)')

    # --- RSI divergence (+3) ---
    rsi_divs = [d for d in divergences if 'RSI' in d.indicators_confirming]
    if rsi_divs:
        strongest = max(rsi_divs, key=lambda x: x.strength)
        if strongest.divergence_type in (DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN):
            score += 3.0
            if direction == SignalDirection.NETRAL:
                direction = SignalDirection.BELI
            triggered.append(f'RSI Divergence Bullish ({strongest.divergence_type.value})')
        elif strongest.divergence_type in (DivergenceType.BEARISH_REGULAR, DivergenceType.BEARISH_HIDDEN):
            score += 3.0
            if direction == SignalDirection.NETRAL:
                direction = SignalDirection.JUAL
            triggered.append(f'RSI Divergence Bearish ({strongest.divergence_type.value})')

    # --- MACD crossover (+2) ---
    macd_line   = _get_val(last, 'MACD_12_26_9')
    macd_signal = _get_val(last, 'MACDs_12_26_9')
    macd_hist   = _get_val(last, 'MACDh_12_26_9')

    if not any(np.isnan(v) for v in [macd_line, macd_signal]) and len(df) >= 2:
        prev = df.iloc[-2]
        prev_macd   = _get_val(prev, 'MACD_12_26_9')
        prev_signal = _get_val(prev, 'MACDs_12_26_9')

        if not any(np.isnan(v) for v in [prev_macd, prev_signal]):
            if prev_macd <= prev_signal and macd_line > macd_signal:
                score += 2.0
                if direction == SignalDirection.NETRAL:
                    direction = SignalDirection.BELI
                triggered.append('MACD Golden Cross')
            elif prev_macd >= prev_signal and macd_line < macd_signal:
                score += 2.0
                if direction == SignalDirection.NETRAL:
                    direction = SignalDirection.JUAL
                triggered.append('MACD Death Cross')

    # --- MACD zero-line cross + histogram momentum (+3) ---
    if not any(np.isnan(v) for v in [macd_line, macd_hist]) and len(df) >= 2:
        prev = df.iloc[-2]
        prev_macd_line = _get_val(prev, 'MACD_12_26_9')
        prev_hist      = _get_val(prev, 'MACDh_12_26_9')

        if not any(np.isnan(v) for v in [prev_macd_line, prev_hist]):
            if prev_macd_line <= 0 and macd_line > 0 and macd_hist > prev_hist:
                score += 3.0
                if direction == SignalDirection.NETRAL:
                    direction = SignalDirection.BELI
                triggered.append('MACD Zero-Line Cross Bullish + Histogram Naik')
            elif prev_macd_line >= 0 and macd_line < 0 and macd_hist < prev_hist:
                score += 3.0
                if direction == SignalDirection.NETRAL:
                    direction = SignalDirection.JUAL
                triggered.append('MACD Zero-Line Cross Bearish + Histogram Turun')

    score = min(score, _DIM_MAX[2])

    return DimensionScore(
        dimension=2,
        name='Momentum',
        score=round(score, 2),
        max_score=_DIM_MAX[2],
        direction=direction,
        triggered_indicators=triggered,
        weight=_DIM_WEIGHTS[2],
    )


# ---------------------------------------------------------------------------
# Task 11.3 — Dimensi 3: Volatilitas (max 6 poin)
# ---------------------------------------------------------------------------

def calculate_dimension3_volatility(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> DimensionScore:
    """
    Dimensi 3 — Volatilitas (max 6 poin):
      +2: Harga di bawah BB lower (bull) atau di atas BB upper (bear)
      +1: BB squeeze terdeteksi (BBB < threshold)
      +3: Harga menyentuh level Fibonacci kunci (38.2%, 50%, 61.8%)
    """
    score = 0.0
    direction = SignalDirection.NETRAL
    triggered: list[str] = []
    last = df.iloc[-1]
    close_price = float(df['Close'].iloc[-1])

    # --- Bollinger Bands (+2 dan +1) ---
    bbl = _get_val_flexible(last, ['BBL_20_2.0', 'BBL_20_2'])
    bbu = _get_val_flexible(last, ['BBU_20_2.0', 'BBU_20_2'])
    bbb = _get_val_flexible(last, ['BBB_20_2.0', 'BBB_20_2'])

    if not any(np.isnan(v) for v in [bbl, bbu]):
        if close_price < bbl:
            score += 2.0
            direction = SignalDirection.BELI
            triggered.append(f'Harga di bawah BB Lower ({close_price:.0f} < {bbl:.0f})')
        elif close_price > bbu:
            score += 2.0
            direction = SignalDirection.JUAL
            triggered.append(f'Harga di atas BB Upper ({close_price:.0f} > {bbu:.0f})')

    if not np.isnan(bbb):
        # BB squeeze: BBB < 5% (bandwidth sempit)
        if bbb < 5.0:
            score += 1.0
            triggered.append(f'BB Squeeze (BBB={bbb:.2f}%)')

    # --- Fibonacci levels (+3) ---
    if len(df) >= 20:
        lookback = df.tail(100)
        swing_high = float(lookback['High'].max())
        swing_low  = float(lookback['Low'].min())
        price_range = swing_high - swing_low

        if price_range > 0:
            key_fibs = [0.382, 0.500, 0.618]
            for ratio in key_fibs:
                fib_level = swing_low + price_range * ratio
                tolerance = price_range * 0.01  # 1% dari range

                if abs(close_price - fib_level) <= tolerance:
                    score += 3.0
                    if direction == SignalDirection.NETRAL:
                        # Jika harga di bawah 50% Fibonacci → potensi support (bull)
                        direction = SignalDirection.BELI if ratio <= 0.5 else SignalDirection.JUAL
                    triggered.append(f'Harga di level Fibonacci {ratio*100:.1f}% ({fib_level:.0f})')
                    break  # Hanya hitung satu level Fibonacci

    score = min(score, _DIM_MAX[3])

    return DimensionScore(
        dimension=3,
        name='Volatilitas',
        score=round(score, 2),
        max_score=_DIM_MAX[3],
        direction=direction,
        triggered_indicators=triggered,
        weight=_DIM_WEIGHTS[3],
    )


# ---------------------------------------------------------------------------
# Task 11.4 — Dimensi 4: Volume (max 7 poin)
# ---------------------------------------------------------------------------

def calculate_dimension4_volume(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> DimensionScore:
    """
    Dimensi 4 — Volume (max 7 poin):
      +3: Volume > 2x rata-rata 20 hari
      +2: OBV trend searah dengan price trend
      +2: MFI < 20 (oversold=bull) atau MFI > 80 (overbought=bear)
    """
    score = 0.0
    direction = SignalDirection.NETRAL
    triggered: list[str] = []
    last = df.iloc[-1]
    close_price = float(df['Close'].iloc[-1])

    # --- Volume spike (+3) ---
    if 'Volume' in df.columns and len(df) >= 20:
        current_vol = float(df['Volume'].iloc[-1])
        avg_vol_20  = float(df['Volume'].iloc[-20:].mean())

        if avg_vol_20 > 0 and current_vol > 2.0 * avg_vol_20:
            score += 3.0
            # Arah volume spike mengikuti arah harga
            if len(df) >= 2:
                prev_close = float(df['Close'].iloc[-2])
                if close_price > prev_close:
                    direction = SignalDirection.BELI
                    triggered.append(f'Volume Spike Bullish ({current_vol/avg_vol_20:.1f}x rata-rata)')
                else:
                    direction = SignalDirection.JUAL
                    triggered.append(f'Volume Spike Bearish ({current_vol/avg_vol_20:.1f}x rata-rata)')
            else:
                triggered.append(f'Volume Spike ({current_vol/avg_vol_20:.1f}x rata-rata)')

    # --- OBV trend (+2) ---
    if 'OBV' in df.columns and len(df) >= 10:
        obv_series = df['OBV'].dropna()
        if len(obv_series) >= 10:
            obv_current = float(obv_series.iloc[-1])
            obv_prev    = float(obv_series.iloc[-10])
            obv_trend_up   = obv_current > obv_prev
            price_trend_up = close_price > float(df['Close'].iloc[-10])

            if obv_trend_up == price_trend_up:
                score += 2.0
                if direction == SignalDirection.NETRAL:
                    direction = SignalDirection.BELI if obv_trend_up else SignalDirection.JUAL
                trend_label = 'Naik' if obv_trend_up else 'Turun'
                triggered.append(f'OBV Konfirmasi Tren {trend_label}')

    # --- MFI extreme (+2) ---
    mfi = _get_val(last, 'MFI_14')
    if not np.isnan(mfi):
        if mfi < 20:
            score += 2.0
            if direction == SignalDirection.NETRAL:
                direction = SignalDirection.BELI
            triggered.append(f'MFI Oversold ({mfi:.1f} < 20)')
        elif mfi > 80:
            score += 2.0
            if direction == SignalDirection.NETRAL:
                direction = SignalDirection.JUAL
            triggered.append(f'MFI Overbought ({mfi:.1f} > 80)')

    score = min(score, _DIM_MAX[4])

    return DimensionScore(
        dimension=4,
        name='Volume',
        score=round(score, 2),
        max_score=_DIM_MAX[4],
        direction=direction,
        triggered_indicators=triggered,
        weight=_DIM_WEIGHTS[4],
    )


# ---------------------------------------------------------------------------
# Task 11.5 — Dimensi 5: Candlestick (max 3 poin)
# ---------------------------------------------------------------------------

def calculate_dimension5_candlestick(
    df: pd.DataFrame,
    data_age: DataAgeClassification,
) -> DimensionScore:
    """
    Dimensi 5 — Pola Candlestick (max 3 poin):
      +3: Pola reversal bullish/bearish (Hammer, Engulfing, dll)
      +1: Pola continuation
    """
    score = 0.0
    direction = SignalDirection.NETRAL
    triggered: list[str] = []
    last = df.iloc[-1]

    # Pola reversal bullish (+3)
    reversal_bullish = {
        'CDL_HAMMER':    'Hammer',
        'CDL_ENGULFING': 'Bullish Engulfing',
    }
    # Pola reversal bearish (+3)
    reversal_bearish = {
        'CDL_SHOOTING_STAR': 'Shooting Star',
    }
    # Pola doji (netral/continuation, +1)
    continuation = {
        'CDL_DOJI_10_0.1': 'Doji',
    }

    for col, name in reversal_bullish.items():
        val = _get_val(last, col)
        if not np.isnan(val) and val > 0:
            score += 3.0
            direction = SignalDirection.BELI
            triggered.append(f'Pola {name} (Bullish Reversal)')
            break

    if score == 0.0:
        for col, name in reversal_bearish.items():
            val = _get_val(last, col)
            if not np.isnan(val) and val < 0:
                score += 3.0
                direction = SignalDirection.JUAL
                triggered.append(f'Pola {name} (Bearish Reversal)')
                break

    if score == 0.0:
        for col, name in continuation.items():
            val = _get_val(last, col)
            if not np.isnan(val) and val != 0:
                score += 1.0
                triggered.append(f'Pola {name} (Continuation)')
                break

    score = min(score, _DIM_MAX[5])

    return DimensionScore(
        dimension=5,
        name='Candlestick',
        score=round(score, 2),
        max_score=_DIM_MAX[5],
        direction=direction,
        triggered_indicators=triggered,
        weight=_DIM_WEIGHTS[5],
    )
