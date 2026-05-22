"""
recommendation/scorer.py — Agregasi Skor dan Filter Kondisi Pasar

Mengagregasi skor dimensi, menerapkan filter regime pasar,
menghitung bonus/penalti keselarasan multi-timeframe,
dan menghitung confidence score akhir.
"""

from __future__ import annotations

import logging

from models import (
    DimensionScore,
    MarketRegime,
    MarketRegimeResult,
    SignalDirection,
    SignalResult,
    SignalStrength,
)

logger = logging.getLogger(__name__)

_TOTAL_MAX = 33.0


# ---------------------------------------------------------------------------
# Task 13.1 — Agregasi skor dimensi
# ---------------------------------------------------------------------------

def aggregate_scores(
    dim_scores: list[DimensionScore],
) -> tuple[float, SignalStrength]:
    """
    Jumlahkan skor semua dimensi dan klasifikasikan kekuatan sinyal.

    Parameters
    ----------
    dim_scores : list[DimensionScore]
        Daftar skor 5 dimensi.

    Returns
    -------
    tuple[float, SignalStrength]
        (total_score, signal_strength)
    """
    total = sum(d.score for d in dim_scores)
    total = max(0.0, min(total, _TOTAL_MAX))

    if total >= 25:
        strength = SignalStrength.STRONG
    elif total >= 17:
        strength = SignalStrength.MODERATE
    elif total >= 10:
        strength = SignalStrength.WEAK
    else:
        strength = SignalStrength.NEUTRAL

    return round(total, 2), strength


# ---------------------------------------------------------------------------
# Task 13.2 — Filter kondisi pasar
# ---------------------------------------------------------------------------

def apply_regime_filter(
    signal: SignalResult,
    regime: MarketRegimeResult,
) -> SignalResult:
    """
    Terapkan filter kondisi pasar pada SignalResult.

    Aturan:
      - BULL_TREND : paksa direction=BELI jika skor cukup (>=10), NETRAL jika tidak
      - BEAR_TREND : paksa direction=JUAL jika skor cukup (>=10), NETRAL jika tidak
      - SIDEWAYS / BREAKOUT : gunakan direction dari skor apa adanya

    Parameters
    ----------
    signal : SignalResult
        Hasil sinyal sebelum filter.
    regime : MarketRegimeResult
        Kondisi pasar yang terdeteksi.

    Returns
    -------
    SignalResult
        SignalResult baru dengan direction yang sudah difilter.
    """
    direction = signal.direction
    total_score = signal.total_score

    if regime.regime == MarketRegime.BULL_TREND:
        if direction == SignalDirection.JUAL:
            # Dalam bull trend, sinyal JUAL diubah ke NETRAL
            direction = SignalDirection.NETRAL
            logger.debug('Regime BULL_TREND: sinyal JUAL diubah ke NETRAL')
        elif direction == SignalDirection.NETRAL and total_score >= 10:
            # Skor cukup dalam bull trend → BELI
            direction = SignalDirection.BELI
            logger.debug('Regime BULL_TREND: sinyal NETRAL dengan skor cukup diubah ke BELI')

    elif regime.regime == MarketRegime.BEAR_TREND:
        if direction == SignalDirection.BELI:
            # Dalam bear trend, sinyal BELI diubah ke NETRAL
            direction = SignalDirection.NETRAL
            logger.debug('Regime BEAR_TREND: sinyal BELI diubah ke NETRAL')
        elif direction == SignalDirection.NETRAL and total_score >= 10:
            # Skor cukup dalam bear trend → JUAL
            direction = SignalDirection.JUAL
            logger.debug('Regime BEAR_TREND: sinyal NETRAL dengan skor cukup diubah ke JUAL')

    # Buat SignalResult baru dengan direction yang diperbarui
    return SignalResult(
        dimension_scores=signal.dimension_scores,
        total_score=signal.total_score,
        signal_strength=signal.signal_strength,
        direction=direction,
        timeframe=signal.timeframe,
        alignment_score=signal.alignment_score,
    )


# ---------------------------------------------------------------------------
# Task 13.3 — Bonus/penalti keselarasan multi-timeframe
# ---------------------------------------------------------------------------

def apply_timeframe_alignment(
    daily: SignalResult,
    weekly: SignalResult,
    monthly: SignalResult,
) -> tuple[SignalResult, SignalResult, SignalResult, float]:
    """
    Hitung bonus/penalti keselarasan timeframe dan perbarui alignment_score.

    Aturan:
      - Ketiga arah sama  : bonus +3.0
      - Dua dari tiga sama: bonus +1.5
      - Semua berbeda     : penalti -2.0

    Parameters
    ----------
    daily, weekly, monthly : SignalResult
        Sinyal untuk masing-masing timeframe.

    Returns
    -------
    tuple[SignalResult, SignalResult, SignalResult, float]
        (daily_updated, weekly_updated, monthly_updated, alignment_bonus)
    """
    dirs = [daily.direction, weekly.direction, monthly.direction]

    beli_count   = dirs.count(SignalDirection.BELI)
    jual_count   = dirs.count(SignalDirection.JUAL)
    netral_count = dirs.count(SignalDirection.NETRAL)

    # Tentukan bonus/penalti
    if beli_count == 3 or jual_count == 3:
        alignment_bonus = 3.0
    elif beli_count == 2 or jual_count == 2:
        alignment_bonus = 1.5
    elif netral_count == 3:
        alignment_bonus = 0.0
    else:
        alignment_bonus = -2.0

    def _update(sig: SignalResult) -> SignalResult:
        new_score = max(0.0, min(sig.total_score + alignment_bonus, _TOTAL_MAX))
        return SignalResult(
            dimension_scores=sig.dimension_scores,
            total_score=round(new_score, 2),
            signal_strength=sig.signal_strength,
            direction=sig.direction,
            timeframe=sig.timeframe,
            alignment_score=alignment_bonus,
        )

    return _update(daily), _update(weekly), _update(monthly), alignment_bonus


# ---------------------------------------------------------------------------
# Task 13.4 — Hitung confidence score
# ---------------------------------------------------------------------------

def calculate_confidence(
    total_score: float,
    max_score: float = _TOTAL_MAX,
) -> float:
    """
    Hitung confidence score dalam persentase.

    confidence = (total_score / max_score) * 100
    Diclamp ke range [0, 100].

    Parameters
    ----------
    total_score : float
        Total skor sinyal (sudah termasuk alignment bonus/penalti).
    max_score : float
        Skor maksimum yang mungkin (default 33).

    Returns
    -------
    float
        Confidence score dalam persen, range [0, 100].
    """
    if max_score <= 0:
        return 0.0
    confidence = (total_score / max_score) * 100.0
    return round(max(0.0, min(confidence, 100.0)), 1)
