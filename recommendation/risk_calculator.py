"""
recommendation/risk_calculator.py — Kalkulator Risiko

Menghitung Stop Loss, Take Profit (TP1/TP2/TP3), ukuran posisi,
dan R/R Ratio berdasarkan aturan risiko 2% per transaksi.

Aturan BEI: 1 lot = 100 lembar saham.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from models import (
    RiskResult,
    SignalDirection,
    SRResult,
)

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.0) -> float:
    """Konversi nilai ke float dengan aman, kembalikan default jika NaN/None."""
    try:
        result = float(value)
        return default if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError):
        return default


def _safe_round(value, ndigits: int = 2) -> float:
    """Round dengan aman, kembalikan 0.0 jika NaN/None."""
    return round(_safe_float(value, 0.0), ndigits)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

_ATR_MULTIPLIER  = 1.5    # SL = entry ± (ATR × 1.5)
_RISK_PCT        = 0.02   # Risiko 2% dari total modal per transaksi
_LOT_SIZE        = 100    # 1 lot BEI = 100 lembar saham
_TP1_RR          = 2.0    # TP1 = SL distance × 2
_TP2_RR          = 3.0    # TP2 = SL distance × 3


# ---------------------------------------------------------------------------
# Task 14.4 — Format Rupiah
# ---------------------------------------------------------------------------

def format_rupiah(value: float) -> str:
    """
    Format nilai float ke format Rupiah Indonesia.

    Contoh: 10000 → 'Rp 10.000'

    Parameters
    ----------
    value : float
        Nilai numerik yang akan diformat.

    Returns
    -------
    str
        String dalam format 'Rp X.XXX'.
    """
    try:
        formatted = f'{value:,.0f}'.replace(',', '.')
        return f'Rp {formatted}'
    except (TypeError, ValueError):
        return 'Rp -'


# ---------------------------------------------------------------------------
# Task 14.5 — Fungsi utama: calculate_risk
# ---------------------------------------------------------------------------

def calculate_risk(
    entry: float,
    direction: SignalDirection,
    atr: float,
    sr_result: SRResult,
    total_capital: float = 0.0,
    data_age=None,
) -> RiskResult:
    """
    Hitung Stop Loss, Take Profit, dan ukuran posisi.

    Parameters
    ----------
    entry : float
        Harga entry (biasanya harga penutupan terakhir).
    direction : SignalDirection
        Arah sinyal: BELI, JUAL, atau NETRAL.
    atr : float
        Nilai ATR terakhir (ATRr_14). Gunakan 0 jika tidak tersedia.
    sr_result : SRResult
        Hasil kalkulasi level Support & Resistance.
    total_capital : float
        Total modal pengguna. 0 = tidak menghitung position sizing.
    data_age : DataAgeClassification, optional
        Klasifikasi usia data. Jika IPO_NEW, skip ATR-based SL.

    Returns
    -------
    RiskResult
    """
    from models import DataAgeClassification

    # Jika sinyal NETRAL, kembalikan hasil minimal
    if direction == SignalDirection.NETRAL:
        return _neutral_risk(entry)

    # Task 14.1 — Hitung Stop Loss
    stop_loss, sl_method = _calculate_stop_loss(
        entry=entry,
        atr=atr,
        sr_result=sr_result,
        direction=direction,
        data_age=data_age,
    )

    # Hitung jarak SL — pastikan stop_loss tidak NaN
    stop_loss = _safe_float(stop_loss, entry * 0.95 if direction == SignalDirection.BELI else entry * 1.05)
    sl_distance = abs(_safe_float(entry, 0.0) - stop_loss)
    sl_distance_pct = (sl_distance / _safe_float(entry, 1.0) * 100) if entry > 0 else 0.0

    # Task 14.2 — Hitung Take Profit
    tp1, tp2, tp3 = _calculate_targets(
        entry=entry,
        sl_distance=sl_distance,
        sr_result=sr_result,
        direction=direction,
    )

    # Task 14.7 — Tangani division by zero pada R/R
    sl_distance_safe = _safe_float(sl_distance, 0.0)
    if sl_distance_safe == 0:
        rr_tp1 = None
        rr_tp2 = None
        rr_tp3 = None
    else:
        def _rr(tp_val: float) -> Optional[float]:
            try:
                v = abs(_safe_float(tp_val, 0.0) - _safe_float(entry, 0.0)) / sl_distance_safe
                return round(v, 2) if not math.isnan(v) and not math.isinf(v) else None
            except Exception:
                return None

        rr_tp1 = _rr(tp1)
        rr_tp2 = _rr(tp2)
        rr_tp3 = _rr(tp3) if _safe_float(tp3, 0.0) != _safe_float(entry, 0.0) else None

    # Task 14.3 — Hitung position sizing
    position_size_lots: Optional[int] = None
    capital_at_risk: Optional[float] = None

    # Task 14.8 — Tangani total_capital = 0
    if total_capital > 0 and sl_distance > 0:
        position_size_lots, capital_at_risk = _calculate_position_size(
            entry=entry,
            stop_loss=stop_loss,
            total_capital=total_capital,
        )

    return RiskResult(
        entry_price=_safe_round(entry),
        stop_loss=_safe_round(stop_loss),
        tp1=_safe_round(tp1),
        tp2=_safe_round(tp2),
        tp3=_safe_round(tp3),
        sl_distance=_safe_round(sl_distance),
        sl_distance_pct=_safe_round(sl_distance_pct),
        rr_ratio_tp1=rr_tp1,
        rr_ratio_tp2=rr_tp2,
        rr_ratio_tp3=rr_tp3,
        position_size_lots=position_size_lots,
        capital_at_risk=capital_at_risk,
        sl_method=sl_method,
    )


# ---------------------------------------------------------------------------
# Task 14.1 — Hitung Stop Loss
# ---------------------------------------------------------------------------

def _calculate_stop_loss(
    entry: float,
    atr: float,
    sr_result: SRResult,
    direction: SignalDirection,
    data_age=None,
) -> tuple[float, str]:
    """
    Pilih Stop Loss yang lebih konservatif antara ATR × 1.5 dan level S/R terdekat.

    Untuk BELI: SL di bawah entry → pilih yang LEBIH TINGGI (lebih konservatif)
    Untuk JUAL: SL di atas entry → pilih yang LEBIH RENDAH (lebih konservatif)

    Returns
    -------
    tuple[float, str]
        (stop_loss_price, metode_yang_digunakan)
    """
    from models import DataAgeClassification

    # Task 14.6 — Cek validitas ATR
    try:
        atr_float = float(atr) if atr is not None else 0.0
        atr_valid = not math.isnan(atr_float) and not math.isinf(atr_float) and atr_float > 0
    except (TypeError, ValueError):
        atr_valid = False

    # Jika IPO_NEW, skip ATR-based SL
    if data_age is not None and data_age == DataAgeClassification.IPO_NEW:
        atr_valid = False

    sl_atr: Optional[float] = None
    sl_sr: Optional[float] = None

    if atr_valid:
        if direction == SignalDirection.BELI:
            sl_atr = entry - (atr_float * _ATR_MULTIPLIER)
        else:
            sl_atr = entry + (atr_float * _ATR_MULTIPLIER)

    # SL berbasis S/R
    if direction == SignalDirection.BELI and sr_result.nearest_support is not None:
        sl_sr = sr_result.nearest_support.price
    elif direction == SignalDirection.JUAL and sr_result.nearest_resistance is not None:
        sl_sr = sr_result.nearest_resistance.price

    # Pilih yang lebih konservatif
    if sl_atr is not None and sl_sr is not None:
        if direction == SignalDirection.BELI:
            if sl_atr < entry and sl_sr < entry:
                stop_loss = max(sl_atr, sl_sr)
                sl_method = 'ATR+S/R (konservatif)'
            elif sl_atr < entry:
                stop_loss = sl_atr
                sl_method = 'ATR'
            elif sl_sr < entry:
                stop_loss = sl_sr
                sl_method = 'S/R'
            else:
                stop_loss = entry * 0.95
                sl_method = 'Fallback 5%'
        else:
            if sl_atr > entry and sl_sr > entry:
                stop_loss = min(sl_atr, sl_sr)
                sl_method = 'ATR+S/R (konservatif)'
            elif sl_atr > entry:
                stop_loss = sl_atr
                sl_method = 'ATR'
            elif sl_sr > entry:
                stop_loss = sl_sr
                sl_method = 'S/R'
            else:
                stop_loss = entry * 1.05
                sl_method = 'Fallback 5%'
    elif sl_atr is not None:
        stop_loss = sl_atr
        sl_method = 'ATR'
    elif sl_sr is not None:
        stop_loss = sl_sr
        sl_method = 'S/R'
    else:
        if direction == SignalDirection.BELI:
            stop_loss = entry * 0.95
        else:
            stop_loss = entry * 1.05
        sl_method = 'Fallback 5%'

    return stop_loss, sl_method


# ---------------------------------------------------------------------------
# Task 14.2 — Hitung Take Profit
# ---------------------------------------------------------------------------

def _calculate_targets(
    entry: float,
    sl_distance: float,
    sr_result: SRResult,
    direction: SignalDirection,
) -> tuple[float, float, float]:
    """
    Hitung TP1, TP2, TP3.

    TP1 = entry ± (sl_distance × 2)
    TP2 = entry ± (sl_distance × 3)
    TP3 = level S/R berikutnya (resistance untuk BELI, support untuk JUAL)
    """
    entry_f = _safe_float(entry, 0.0)
    sl_dist = _safe_float(sl_distance, 0.0)

    if direction == SignalDirection.BELI:
        tp1 = entry_f + sl_dist * _TP1_RR
        tp2 = entry_f + sl_dist * _TP2_RR

        if sr_result.nearest_resistance is not None:
            tp3_candidate = _safe_float(sr_result.nearest_resistance.price, 0.0)
            tp3 = tp3_candidate if tp3_candidate > tp2 else tp2 * 1.05
        else:
            tp3 = tp2 * 1.05

    else:  # JUAL
        tp1 = entry_f - sl_dist * _TP1_RR
        tp2 = entry_f - sl_dist * _TP2_RR

        if sr_result.nearest_support is not None:
            tp3_candidate = _safe_float(sr_result.nearest_support.price, 0.0)
            tp3 = tp3_candidate if tp3_candidate < tp2 else tp2 * 0.95
        else:
            tp3 = tp2 * 0.95

    # Pastikan tidak ada NaN/Inf di output
    tp1 = _safe_float(tp1, entry_f)
    tp2 = _safe_float(tp2, entry_f)
    tp3 = _safe_float(tp3, entry_f)

    return tp1, tp2, tp3


# ---------------------------------------------------------------------------
# Task 14.3 — Hitung Position Size
# ---------------------------------------------------------------------------

def _calculate_position_size(
    entry: float,
    stop_loss: float,
    total_capital: float,
) -> tuple[int, float]:
    """
    Hitung ukuran posisi berdasarkan aturan risiko 2%.

    risk_amount    = total_capital × 2%
    risk_per_share = |entry - stop_loss|
    shares         = risk_amount / risk_per_share
    lots           = int(shares / 100)  # 1 lot = 100 lembar

    Returns
    -------
    tuple[int, float]
        (jumlah_lot, modal_yang_digunakan)
    """
    try:
        entry_f      = _safe_float(entry, 0.0)
        stop_loss_f  = _safe_float(stop_loss, 0.0)
        capital_f    = _safe_float(total_capital, 0.0)

        if entry_f <= 0 or capital_f <= 0:
            return 0, 0.0

        risk_per_share = abs(entry_f - stop_loss_f)
        if risk_per_share <= 0:
            return 0, 0.0

        risk_amount = capital_f * _RISK_PCT
        shares_raw  = risk_amount / risk_per_share

        # Pastikan shares_raw bukan NaN/Inf sebelum konversi ke int
        if math.isnan(shares_raw) or math.isinf(shares_raw) or shares_raw < 0:
            return 0, 0.0

        lots = int(shares_raw / _LOT_SIZE)
        if lots <= 0:
            return 0, 0.0

        capital_used = lots * _LOT_SIZE * entry_f
        return lots, round(capital_used, 2)

    except Exception as e:
        logger.warning(f'Gagal menghitung position size: {e}')
        return 0, 0.0


# ---------------------------------------------------------------------------
# Helper: neutral risk result
# ---------------------------------------------------------------------------

def _neutral_risk(entry: float) -> RiskResult:
    """Kembalikan RiskResult kosong untuk sinyal NETRAL."""
    return RiskResult(
        entry_price=round(entry, 2),
        stop_loss=0.0,
        tp1=0.0,
        tp2=0.0,
        tp3=0.0,
        sl_distance=0.0,
        sl_distance_pct=0.0,
        rr_ratio_tp1=None,
        rr_ratio_tp2=None,
        rr_ratio_tp3=None,
        position_size_lots=None,
        capital_at_risk=None,
        sl_method='N/A',
    )
