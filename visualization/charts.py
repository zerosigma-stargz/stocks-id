"""
visualization/charts.py — Visualisasi Engine

Membangun grafik teknikal interaktif menggunakan Plotly dengan 4 panel:
  Row 1 (50%): Candlestick + MA + BB + Fibonacci + S/R + sinyal
  Row 2 (18%): RSI + divergensi
  Row 3 (18%): MACD + histogram
  Row 4 (14%): Volume + OBV + rata-rata volume
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import (
    DataAgeClassification,
    DivergenceResult,
    DivergenceType,
    SignalDirection,
    SignalResult,
    SRLevelType,
    SRResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: pencarian kolom fleksibel
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Cari kolom pertama dari daftar kandidat yang ada di df dan tidak semua NaN.

    Parameters
    ----------
    df : pd.DataFrame
    candidates : list[str]
        Daftar nama kolom yang mungkin, diurutkan dari prioritas tertinggi.

    Returns
    -------
    str atau None
    """
    # Cek eksak dulu
    for col in candidates:
        if col in df.columns and not df[col].isna().all():
            return col

    # Fallback: cari berdasarkan prefix dari kandidat pertama
    if candidates:
        # Ambil prefix sebelum underscore pertama (misal 'BBL' dari 'BBL_20_2.0')
        prefix = candidates[0].split('_')[0]
        for col in df.columns:
            if col.startswith(prefix + '_') and not df[col].isna().all():
                return col
    return None


# ---------------------------------------------------------------------------
# Konstanta warna
# ---------------------------------------------------------------------------

_COLOR_BELI    = '#00C851'
_COLOR_JUAL    = '#FF4444'
_COLOR_NETRAL  = '#9E9E9E'
_COLOR_MA20    = '#2196F3'   # Biru
_COLOR_MA50    = '#FF9800'   # Oranye
_COLOR_MA200   = '#F44336'   # Merah
_COLOR_BB      = 'rgba(158, 158, 158, 0.3)'
_COLOR_SUPPORT = 'rgba(0, 200, 81, 0.15)'
_COLOR_RESIST  = 'rgba(255, 68, 68, 0.15)'
_COLOR_FIB     = 'rgba(255, 193, 7, 0.6)'
_COLOR_OBV     = '#9C27B0'   # Ungu


# ---------------------------------------------------------------------------
# Task 17.7 — Fungsi utama: build_combined_chart
# ---------------------------------------------------------------------------

def build_combined_chart(
    df: pd.DataFrame,
    sr_result: Optional[SRResult] = None,
    signal_daily: Optional[SignalResult] = None,
    divergences: Optional[list[DivergenceResult]] = None,
    data_age: Optional[DataAgeClassification] = None,
) -> go.Figure:
    """
    Bangun grafik teknikal interaktif dengan 4 panel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame OHLCV harian dengan kolom indikator yang sudah dihitung.
    sr_result : SRResult, optional
        Level Support & Resistance.
    signal_daily : SignalResult, optional
        Sinyal harian untuk marker BELI/JUAL.
    divergences : list[DivergenceResult], optional
        Daftar divergensi untuk anotasi di panel RSI.
    data_age : DataAgeClassification, optional
        Klasifikasi usia data.

    Returns
    -------
    go.Figure
        Plotly figure siap ditampilkan.
    """
    if df is None or df.empty:
        return go.Figure()

    divergences = divergences or []

    # Task 17.1 — Setup subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        subplot_titles=('', 'RSI', 'MACD', 'Volume'),
    )

    # Task 17.2 — Row 1: Candlestick + MA + BB
    _add_candlestick(fig, df)
    _add_moving_averages(fig, df)
    _add_bollinger_bands(fig, df)

    # Task 17.3 — Row 1: Fibonacci + S/R + sinyal
    _add_fibonacci_levels(fig, df)
    if sr_result is not None:
        _add_sr_zones(fig, sr_result)
    if signal_daily is not None:
        _add_signal_markers(fig, df, signal_daily)

    # Task 17.4 — Row 2: RSI + divergensi
    _add_rsi_panel(fig, df, divergences)

    # Task 17.5 — Row 3: MACD + histogram
    _add_macd_panel(fig, df)

    # Task 17.6 — Row 4: Volume + OBV + rata-rata
    _add_volume_panel(fig, df)

    # Layout umum
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FAFAFA', size=11),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    # Warna axis
    for i in range(1, 5):
        fig.update_xaxes(
            row=i, col=1,
            gridcolor='#1E2130',
            zerolinecolor='#1E2130',
            showgrid=True,
        )
        fig.update_yaxes(
            row=i, col=1,
            gridcolor='#1E2130',
            zerolinecolor='#1E2130',
            showgrid=True,
        )

    return fig


# ---------------------------------------------------------------------------
# Task 17.2 — Candlestick
# ---------------------------------------------------------------------------

def _add_candlestick(fig: go.Figure, df: pd.DataFrame) -> None:
    """Tambahkan candlestick chart ke Row 1."""
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color=_COLOR_BELI,
            decreasing_line_color=_COLOR_JUAL,
            increasing_fillcolor=_COLOR_BELI,
            decreasing_fillcolor=_COLOR_JUAL,
        ),
        row=1, col=1,
    )


def _add_moving_averages(fig: go.Figure, df: pd.DataFrame) -> None:
    """Tambahkan MA20, MA50, MA200 ke Row 1."""
    ma_config = [
        ('SMA_20',  'MA20',  _COLOR_MA20,  1.5),
        ('SMA_50',  'MA50',  _COLOR_MA50,  1.5),
        ('SMA_200', 'MA200', _COLOR_MA200, 2.0),
    ]
    for col, name, color, width in ma_config:
        if col in df.columns and not df[col].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=name,
                    line=dict(color=color, width=width),
                    opacity=0.85,
                ),
                row=1, col=1,
            )


def _add_bollinger_bands(fig: go.Figure, df: pd.DataFrame) -> None:
    """Tambahkan Bollinger Bands ke Row 1 dengan fill antara upper dan lower."""
    # Cari kolom BB secara fleksibel (BBL_20_2.0 atau BBL_20_2, dll)
    bbl_col = _find_col(df, ['BBL_20_2.0', 'BBL_20_2'])
    bbu_col = _find_col(df, ['BBU_20_2.0', 'BBU_20_2'])
    bbm_col = _find_col(df, ['BBM_20_2.0', 'BBM_20_2'])

    if bbl_col is None or df[bbl_col].isna().all():
        return
    if bbu_col is None or df[bbu_col].isna().all():
        return

    # Upper band
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[bbu_col],
            name='BB Upper',
            line=dict(color='rgba(158,158,158,0.5)', width=1, dash='dot'),
            showlegend=False,
        ),
        row=1, col=1,
    )
    # Lower band dengan fill ke upper
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[bbl_col],
            name='Bollinger Bands',
            line=dict(color='rgba(158,158,158,0.5)', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(158,158,158,0.08)',
        ),
        row=1, col=1,
    )
    # Middle band
    if bbm_col is not None and not df[bbm_col].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[bbm_col],
                name='BB Middle',
                line=dict(color='rgba(158,158,158,0.4)', width=1),
                showlegend=False,
            ),
            row=1, col=1,
        )


# ---------------------------------------------------------------------------
# Task 17.3 — Fibonacci + S/R + sinyal
# ---------------------------------------------------------------------------

def _add_fibonacci_levels(fig: go.Figure, df: pd.DataFrame) -> None:
    """Tambahkan garis horizontal Fibonacci ke Row 1."""
    if len(df) < 20:
        return

    lookback = df.tail(100)
    swing_high = float(lookback['High'].max())
    swing_low  = float(lookback['Low'].min())
    price_range = swing_high - swing_low

    if price_range <= 0:
        return

    fib_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    for ratio in fib_ratios:
        fib_price = swing_low + price_range * ratio
        fig.add_hline(
            y=fib_price,
            line=dict(color=_COLOR_FIB, width=1, dash='dash'),
            annotation_text=f'Fib {ratio*100:.1f}%',
            annotation_position='right',
            annotation_font=dict(size=9, color=_COLOR_FIB),
            row=1, col=1,
        )


def _add_sr_zones(fig: go.Figure, sr_result: SRResult) -> None:
    """Tambahkan zona S/R sebagai area berwarna semi-transparan ke Row 1."""
    if not sr_result.all_levels:
        return

    current_price = sr_result.current_price
    atr = sr_result.atr if sr_result.atr > 0 else current_price * 0.01

    for level in sr_result.all_levels:
        color = _COLOR_SUPPORT if level.level_type == SRLevelType.SUPPORT else _COLOR_RESIST
        # Zona: ±0.5 ATR di sekitar level
        y0 = level.price - atr * 0.3
        y1 = level.price + atr * 0.3

        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            annotation_text=f'{level.source.value[:3]} {level.price:.0f}',
            annotation_position='right',
            annotation_font=dict(size=8),
            row=1, col=1,
        )


def _add_signal_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    signal: SignalResult,
) -> None:
    """Tambahkan marker BELI (triangle-up) atau JUAL (triangle-down) ke Row 1."""
    if signal.direction == SignalDirection.NETRAL:
        return

    last_idx = df.index[-1]
    last_close = float(df['Close'].iloc[-1])

    if signal.direction == SignalDirection.BELI:
        fig.add_trace(
            go.Scatter(
                x=[last_idx],
                y=[last_close * 0.98],
                mode='markers',
                name=f'Sinyal BELI ({signal.signal_strength.value})',
                marker=dict(
                    symbol='triangle-up',
                    size=16,
                    color=_COLOR_BELI,
                    line=dict(color='white', width=1),
                ),
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[last_idx],
                y=[last_close * 1.02],
                mode='markers',
                name=f'Sinyal JUAL ({signal.signal_strength.value})',
                marker=dict(
                    symbol='triangle-down',
                    size=16,
                    color=_COLOR_JUAL,
                    line=dict(color='white', width=1),
                ),
            ),
            row=1, col=1,
        )


# ---------------------------------------------------------------------------
# Task 17.4 — Panel RSI
# ---------------------------------------------------------------------------

def _add_rsi_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    divergences: list[DivergenceResult],
) -> None:
    """Tambahkan RSI dan anotasi divergensi ke Row 2."""
    if 'RSI_14' not in df.columns or df['RSI_14'].isna().all():
        return

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI_14'],
            name='RSI(14)',
            line=dict(color='#E91E63', width=1.5),
        ),
        row=2, col=1,
    )

    # Garis overbought (70) dan oversold (30)
    fig.add_hline(y=70, line=dict(color=_COLOR_JUAL, width=1, dash='dot'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=_COLOR_BELI, width=1, dash='dot'), row=2, col=1)
    fig.add_hline(y=50, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), row=2, col=1)

    # Anotasi divergensi RSI
    rsi_divs = [d for d in divergences if 'RSI' in d.indicators_confirming]
    for div in rsi_divs[:3]:  # Maksimal 3 divergensi
        if len(div.indicator_pivot_indices) < 2:
            continue
        try:
            idx0 = div.indicator_pivot_indices[0]
            idx1 = div.indicator_pivot_indices[1]
            if idx0 >= len(df) or idx1 >= len(df):
                continue

            x0 = df.index[idx0]
            x1 = df.index[idx1]
            y0 = div.indicator_pivot_values[0]
            y1 = div.indicator_pivot_values[1]

            color = _COLOR_BELI if 'BULLISH' in div.divergence_type.value else _COLOR_JUAL

            fig.add_shape(
                type='line',
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=2, dash='dot'),
                row=2, col=1,
            )
        except (IndexError, KeyError):
            continue

    fig.update_yaxes(range=[0, 100], row=2, col=1)


# ---------------------------------------------------------------------------
# Task 17.5 — Panel MACD
# ---------------------------------------------------------------------------

def _add_macd_panel(fig: go.Figure, df: pd.DataFrame) -> None:
    """Tambahkan MACD line, signal line, dan histogram ke Row 3."""
    macd_col  = 'MACD_12_26_9'
    sig_col   = 'MACDs_12_26_9'
    hist_col  = 'MACDh_12_26_9'

    if macd_col not in df.columns or df[macd_col].isna().all():
        return

    # Histogram (hijau jika positif, merah jika negatif)
    if hist_col in df.columns and not df[hist_col].isna().all():
        hist = df[hist_col].fillna(0)
        colors = [_COLOR_BELI if v >= 0 else _COLOR_JUAL for v in hist]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=hist,
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.7,
            ),
            row=3, col=1,
        )

    # MACD line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[macd_col],
            name='MACD',
            line=dict(color=_COLOR_MA20, width=1.5),
        ),
        row=3, col=1,
    )

    # Signal line
    if sig_col in df.columns and not df[sig_col].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[sig_col],
                name='Signal',
                line=dict(color=_COLOR_MA50, width=1.5),
            ),
            row=3, col=1,
        )

    # Zero line
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1), row=3, col=1)


# ---------------------------------------------------------------------------
# Task 17.6 — Panel Volume
# ---------------------------------------------------------------------------

def _add_volume_panel(fig: go.Figure, df: pd.DataFrame) -> None:
    """Tambahkan volume bar, OBV overlay, dan rata-rata volume 20 hari ke Row 4."""
    if 'Volume' not in df.columns:
        return

    # Warna bar: hijau jika close > open, merah jika sebaliknya
    vol_colors = [
        _COLOR_BELI if float(df['Close'].iloc[i]) >= float(df['Open'].iloc[i])
        else _COLOR_JUAL
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=vol_colors,
            opacity=0.6,
        ),
        row=4, col=1,
    )

    # Rata-rata volume 20 hari
    if len(df) >= 20:
        avg_vol = df['Volume'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=avg_vol,
                name='Avg Vol 20',
                line=dict(color='rgba(255,255,255,0.5)', width=1.5, dash='dash'),
            ),
            row=4, col=1,
        )

    # OBV (secondary y-axis simulasi — tampilkan sebagai overlay ternormalisasi)
    if 'OBV' in df.columns and not df['OBV'].isna().all():
        obv = df['OBV'].dropna()
        if len(obv) > 0:
            # Normalisasi OBV ke skala volume untuk overlay
            obv_min = obv.min()
            obv_max = obv.max()
            vol_max = df['Volume'].max()

            if obv_max != obv_min and vol_max > 0:
                obv_normalized = (obv - obv_min) / (obv_max - obv_min) * vol_max * 0.8
                fig.add_trace(
                    go.Scatter(
                        x=obv.index,
                        y=obv_normalized,
                        name='OBV (norm.)',
                        line=dict(color=_COLOR_OBV, width=1.5),
                        opacity=0.8,
                    ),
                    row=4, col=1,
                )
