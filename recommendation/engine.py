"""
recommendation/engine.py — Mesin Rekomendasi

Mengorkestrasikan semua modul untuk menghasilkan:
  - 3 kartu rekomendasi (jangka pendek, menengah, panjang)
  - Narasi interpretasi gabungan dalam Bahasa Indonesia
  - AnalysisResult lengkap
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from models import (
    AnalysisResult,
    DataAgeClassification,
    RecommendationCard,
    RiskResult,
    SignalDirection,
    SignalResult,
    SignalStrength,
    SRResult,
)
from recommendation.scorer import (
    aggregate_scores,
    apply_regime_filter,
    apply_timeframe_alignment,
    calculate_confidence,
)
from recommendation.risk_calculator import calculate_risk, format_rupiah

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta horizon investasi
# ---------------------------------------------------------------------------

_HORIZONS = [
    {
        'key':          'pendek',
        'label':        'Jangka Pendek',
        'detail':       '1–3 Bulan',
        'timeframe':    'daily',
    },
    {
        'key':          'menengah',
        'label':        'Jangka Menengah',
        'detail':       '3–12 Bulan',
        'timeframe':    'weekly',
    },
    {
        'key':          'panjang',
        'label':        'Jangka Panjang',
        'detail':       '1–3 Tahun',
        'timeframe':    'monthly',
    },
]


# ---------------------------------------------------------------------------
# Task 15.5 — Orkestrator utama: run_full_analysis
# ---------------------------------------------------------------------------

def run_full_analysis(
    ticker: str,
    start_date: str,
    end_date: str,
    total_capital: float = 0.0,
) -> AnalysisResult:
    """
    Orkestrasikan semua modul secara berurutan untuk menghasilkan AnalysisResult.

    Langkah-langkah:
      1.  Ambil data OHLCV multi-timeframe
      2.  Klasifikasi usia data
      3.  Ambil info saham (nama, sektor, harga)
      4.  Deteksi kondisi pasar (market regime)
      5.  Hitung indikator teknikal (daily, weekly, monthly)
      6.  Deteksi divergensi
      7.  Hitung level S/R
      8.  Hitung sinyal 5 dimensi (daily, weekly, monthly)
      9.  Terapkan regime filter
      10. Terapkan timeframe alignment
      11. Hitung risiko (SL, TP, position size)
      12. Generate 3 kartu rekomendasi
      13. Generate narasi gabungan
      14. Susun AnalysisResult

    Parameters
    ----------
    ticker : str
        Kode saham BEI, mis. 'BBCA.JK' atau '^JKSE'.
    start_date : str
        Tanggal mulai format 'YYYY-MM-DD'.
    end_date : str
        Tanggal akhir format 'YYYY-MM-DD'.
    total_capital : float
        Total modal pengguna untuk position sizing. 0 = tidak dihitung.

    Returns
    -------
    AnalysisResult
    """
    from data.fetcher import classify_data_age, fetch_all_timeframes, get_stock_info
    from analysis.indicators import calculate_all_indicators
    from analysis.divergence import detect_divergences
    from analysis.support_resistance import calculate_sr_levels
    from analysis.market_regime import detect_market_regime
    from analysis.signals import calculate_signal

    # Langkah 1 — Ambil data OHLCV
    timeframes = fetch_all_timeframes(ticker, start_date, end_date)
    df_daily   = timeframes['1d']
    df_weekly  = timeframes['1wk']
    df_monthly = timeframes['1mo']

    # Langkah 2 — Klasifikasi usia data
    data_age = classify_data_age(df_daily)
    days_available = len(df_daily)

    # Langkah 3 — Info saham
    try:
        stock_info = get_stock_info(ticker)
        company_name  = stock_info.get('name', ticker)
        current_price = float(df_daily['Close'].iloc[-1])
    except Exception:
        company_name  = ticker
        current_price = float(df_daily['Close'].iloc[-1])

    # Hitung perubahan harga harian
    if len(df_daily) >= 2:
        prev_close = float(df_daily['Close'].iloc[-2])
        price_change_pct = (current_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
    else:
        price_change_pct = 0.0

    # Langkah 4 — Deteksi kondisi pasar (dari data harian)
    market_regime = detect_market_regime(df_daily)

    # Langkah 5 — Hitung indikator teknikal
    df_daily_ind   = calculate_all_indicators(df_daily,   data_age)
    df_weekly_ind  = calculate_all_indicators(df_weekly,  data_age)
    df_monthly_ind = calculate_all_indicators(df_monthly, data_age)

    # Langkah 6 — Deteksi divergensi (dari data harian)
    divergences = detect_divergences(df_daily_ind)

    # Langkah 7 — Hitung level S/R (dari data harian)
    sr_result = calculate_sr_levels(df_daily_ind, data_age)

    # Langkah 8 — Hitung sinyal 5 dimensi
    signal_daily   = calculate_signal(df_daily_ind,   divergences, data_age, market_regime, 'daily')
    signal_weekly  = calculate_signal(df_weekly_ind,  divergences, data_age, market_regime, 'weekly')
    signal_monthly = calculate_signal(df_monthly_ind, divergences, data_age, market_regime, 'monthly')

    # Langkah 9 — Terapkan regime filter
    signal_daily   = apply_regime_filter(signal_daily,   market_regime)
    signal_weekly  = apply_regime_filter(signal_weekly,  market_regime)
    signal_monthly = apply_regime_filter(signal_monthly, market_regime)

    # Langkah 10 — Terapkan timeframe alignment
    signal_daily, signal_weekly, signal_monthly, alignment_bonus = apply_timeframe_alignment(
        signal_daily, signal_weekly, signal_monthly
    )

    # Langkah 11 — Hitung risiko
    risk_result = calculate_risk(
        entry=current_price,
        direction=signal_daily.direction,
        atr=sr_result.atr,
        sr_result=sr_result,
        total_capital=total_capital,
        data_age=data_age,
    )

    # Langkah 12 — Generate 3 kartu rekomendasi
    signal_map = {
        'daily':   signal_daily,
        'weekly':  signal_weekly,
        'monthly': signal_monthly,
    }
    alignment_flag = abs(alignment_bonus) >= 3.0

    recommendations = generate_recommendations(
        signals=signal_map,
        risk=risk_result,
        regime=market_regime,
        alignment=alignment_flag,
    )

    # Langkah 13 — Generate narasi gabungan
    combined_narrative = _generate_narrative(recommendations, market_regime, ticker)

    # Langkah 14 — Susun AnalysisResult
    return AnalysisResult(
        ticker=ticker,
        company_name=company_name,
        current_price=current_price,
        price_change_pct=round(price_change_pct, 2),
        data_age=data_age,
        days_available=days_available,
        market_regime=market_regime,
        signal_daily=signal_daily,
        signal_weekly=signal_weekly,
        signal_monthly=signal_monthly,
        divergences=divergences,
        sr_result=sr_result,
        risk_result=risk_result,
        recommendations=recommendations,
        combined_narrative=combined_narrative,
        analysis_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )


# ---------------------------------------------------------------------------
# Task 15.4 — Generate 3 kartu rekomendasi
# ---------------------------------------------------------------------------

def generate_recommendations(
    signals: dict[str, SignalResult],
    risk: RiskResult,
    regime,
    alignment: bool,
) -> list[RecommendationCard]:
    """
    Buat 3 kartu rekomendasi: pendek (daily), menengah (weekly), panjang (monthly).

    Parameters
    ----------
    signals : dict[str, SignalResult]
        Kunci: 'daily', 'weekly', 'monthly'.
    risk : RiskResult
        Hasil kalkulasi risiko.
    regime : MarketRegimeResult
        Kondisi pasar.
    alignment : bool
        True jika ketiga timeframe selaras.

    Returns
    -------
    list[RecommendationCard]
        Tiga kartu rekomendasi.
    """
    cards: list[RecommendationCard] = []

    for horizon_info in _HORIZONS:
        tf_key = horizon_info['timeframe']
        signal = signals.get(tf_key)
        if signal is None:
            continue

        card = _generate_card(
            horizon=horizon_info['label'],
            horizon_detail=horizon_info['detail'],
            signal=signal,
            risk=risk,
            alignment=alignment,
        )
        cards.append(card)

    return cards


# ---------------------------------------------------------------------------
# Task 15.1 — Generate satu kartu rekomendasi
# ---------------------------------------------------------------------------

def _generate_card(
    horizon: str,
    horizon_detail: str,
    signal: SignalResult,
    risk: RiskResult,
    alignment: bool,
) -> RecommendationCard:
    """Buat satu RecommendationCard dari sinyal dan data risiko."""
    confidence = calculate_confidence(signal.total_score)

    holding_duration = _determine_holding_duration(horizon, signal.signal_strength)

    # Kumpulkan penjelasan dari indikator yang terpicu
    explanation: list[str] = []
    for dim in signal.dimension_scores:
        for ind in dim.triggered_indicators:
            explanation.append(f'[{dim.name}] {ind}')

    if not explanation:
        explanation = ['Belum ada sinyal yang cukup kuat saat ini.']

    return RecommendationCard(
        horizon=horizon,
        horizon_detail=horizon_detail,
        signal=signal.direction,
        signal_strength=signal.signal_strength,
        confidence_pct=confidence,
        entry_price=risk.entry_price,
        stop_loss=risk.stop_loss,
        tp1=risk.tp1,
        tp2=risk.tp2,
        tp3=risk.tp3,
        rr_ratio=risk.rr_ratio_tp1,
        holding_duration=holding_duration,
        explanation=explanation,
        timeframe_alignment=alignment,
        based_on_timeframe=signal.timeframe,
    )


# ---------------------------------------------------------------------------
# Task 15.2 — Tentukan durasi holding
# ---------------------------------------------------------------------------

def _determine_holding_duration(horizon: str, strength: SignalStrength) -> str:
    """
    Tentukan durasi holding yang disarankan berdasarkan horizon dan kekuatan sinyal.

    Returns
    -------
    str
        Contoh: '2–4 minggu', '3–6 bulan', '1–2 tahun'
    """
    duration_map = {
        ('Jangka Pendek',   SignalStrength.STRONG):   '2–4 minggu',
        ('Jangka Pendek',   SignalStrength.MODERATE): '1–2 minggu',
        ('Jangka Pendek',   SignalStrength.WEAK):     '3–7 hari',
        ('Jangka Pendek',   SignalStrength.NEUTRAL):  'Pantau dulu',
        ('Jangka Menengah', SignalStrength.STRONG):   '6–12 bulan',
        ('Jangka Menengah', SignalStrength.MODERATE): '3–6 bulan',
        ('Jangka Menengah', SignalStrength.WEAK):     '1–3 bulan',
        ('Jangka Menengah', SignalStrength.NEUTRAL):  'Pantau dulu',
        ('Jangka Panjang',  SignalStrength.STRONG):   '2–3 tahun',
        ('Jangka Panjang',  SignalStrength.MODERATE): '1–2 tahun',
        ('Jangka Panjang',  SignalStrength.WEAK):     '6–12 bulan',
        ('Jangka Panjang',  SignalStrength.NEUTRAL):  'Pantau dulu',
    }
    return duration_map.get((horizon, strength), 'Pantau dulu')


# ---------------------------------------------------------------------------
# Task 15.3 — Generate narasi gabungan
# ---------------------------------------------------------------------------

def _generate_narrative(
    cards: list[RecommendationCard],
    regime,
    ticker: str,
) -> str:
    """
    Buat narasi interpretasi gabungan dalam Bahasa Indonesia.

    Template berdasarkan kombinasi sinyal 3 horizon.
    """
    if not cards or len(cards) < 3:
        return f'Analisis {ticker} belum dapat menghasilkan narasi lengkap.'

    pendek   = cards[0]
    menengah = cards[1]
    panjang  = cards[2]

    regime_label = regime.regime.value.replace('_', ' ').title()

    # Semua NETRAL
    if all(c.signal == SignalDirection.NETRAL for c in cards):
        return (
            f'Kondisi pasar {ticker} saat ini berada dalam fase {regime_label}. '
            'Belum ada sinyal yang cukup kuat dari ketiga horizon investasi. '
            'Disarankan untuk memantau perkembangan pasar sebelum mengambil posisi.'
        )

    # Semua BELI
    if all(c.signal == SignalDirection.BELI for c in cards):
        return (
            f'{ticker} menunjukkan sinyal BELI yang kuat di semua horizon investasi '
            f'dalam kondisi pasar {regime_label}. '
            f'Jangka pendek ({pendek.confidence_pct:.0f}%), '
            f'menengah ({menengah.confidence_pct:.0f}%), dan '
            f'panjang ({panjang.confidence_pct:.0f}%) semuanya mengonfirmasi momentum naik. '
            'Keselarasan multi-timeframe meningkatkan keyakinan sinyal ini.'
        )

    # Semua JUAL
    if all(c.signal == SignalDirection.JUAL for c in cards):
        return (
            f'{ticker} menunjukkan sinyal JUAL yang kuat di semua horizon investasi '
            f'dalam kondisi pasar {regime_label}. '
            f'Jangka pendek ({pendek.confidence_pct:.0f}%), '
            f'menengah ({menengah.confidence_pct:.0f}%), dan '
            f'panjang ({panjang.confidence_pct:.0f}%) semuanya mengonfirmasi tekanan jual. '
            'Pertimbangkan untuk mengurangi eksposur atau memasang stop loss yang ketat.'
        )

    # Panjang BELI, pendek JUAL (koreksi dalam uptrend)
    if panjang.signal == SignalDirection.BELI and pendek.signal == SignalDirection.JUAL:
        return (
            f'{ticker} berada dalam tren naik jangka panjang ({panjang.confidence_pct:.0f}%) '
            f'namun mengalami koreksi jangka pendek ({pendek.confidence_pct:.0f}%). '
            f'Kondisi pasar: {regime_label}. '
            'Ini bisa menjadi peluang akumulasi bagi investor jangka panjang. '
            'Tunggu konfirmasi pembalikan di timeframe pendek sebelum masuk posisi.'
        )

    # Panjang JUAL, pendek BELI (rebound dalam downtrend)
    if panjang.signal == SignalDirection.JUAL and pendek.signal == SignalDirection.BELI:
        return (
            f'{ticker} berada dalam tren turun jangka panjang ({panjang.confidence_pct:.0f}%) '
            f'namun menunjukkan rebound jangka pendek ({pendek.confidence_pct:.0f}%). '
            f'Kondisi pasar: {regime_label}. '
            'Rebound ini mungkin bersifat sementara. '
            'Berhati-hati dengan sinyal beli jangka pendek yang berlawanan dengan tren utama.'
        )

    # Narasi umum untuk kombinasi lainnya
    signals_text = ', '.join([
        f'{c.horizon}: {c.signal.value} ({c.confidence_pct:.0f}%)'
        for c in cards
    ])
    return (
        f'Analisis {ticker} dalam kondisi pasar {regime_label}: {signals_text}. '
        'Perhatikan keselarasan antar timeframe sebelum mengambil keputusan investasi. '
        'Selalu gunakan stop loss untuk membatasi risiko.'
    )


# ---------------------------------------------------------------------------
# Task 15.6 — Tangani semua sinyal NETRAL
# ---------------------------------------------------------------------------

def all_signals_neutral(cards: list[RecommendationCard]) -> bool:
    """
    Periksa apakah semua kartu rekomendasi memiliki sinyal NETRAL.

    Digunakan oleh UI untuk menampilkan pesan khusus.

    Returns
    -------
    bool
        True jika semua kartu NETRAL.
    """
    if not cards:
        return True
    return all(c.signal == SignalDirection.NETRAL for c in cards)
