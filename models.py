"""
models.py - Data models untuk StockMomentum ID

Berisi semua enum dan dataclass yang digunakan di seluruh aplikasi.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd


# ---------------------------------------------------------------------------
# Enumerasi
# ---------------------------------------------------------------------------

class DataAgeClassification(Enum):
    IPO_NEW = 'IPO_NEW'
    IPO_PARTIAL = 'IPO_PARTIAL'
    STANDARD = 'STANDARD'
    FULL = 'FULL'


class MarketRegime(Enum):
    BULL_TREND = 'BULL_TREND'
    BEAR_TREND = 'BEAR_TREND'
    SIDEWAYS = 'SIDEWAYS'
    BREAKOUT = 'BREAKOUT'


class SignalDirection(Enum):
    BELI = 'BELI'
    JUAL = 'JUAL'
    NETRAL = 'NETRAL'


class SignalStrength(Enum):
    STRONG = 'STRONG'
    MODERATE = 'MODERATE'
    WEAK = 'WEAK'
    NEUTRAL = 'NEUTRAL'


class DivergenceType(Enum):
    BULLISH_REGULAR = 'BULLISH_REGULAR'
    BEARISH_REGULAR = 'BEARISH_REGULAR'
    BULLISH_HIDDEN = 'BULLISH_HIDDEN'
    BEARISH_HIDDEN = 'BEARISH_HIDDEN'


class SRLevelType(Enum):
    SUPPORT = 'SUPPORT'
    RESISTANCE = 'RESISTANCE'


class SRLevelSource(Enum):
    STATIC = 'STATIC'
    DYNAMIC_MA = 'DYNAMIC_MA'
    FIBONACCI = 'FIBONACCI'
    PSYCHOLOGICAL = 'PSYCHOLOGICAL'
    ICHIMOKU = 'ICHIMOKU'


# ---------------------------------------------------------------------------
# Dataclass - Data OHLCV
# ---------------------------------------------------------------------------

@dataclass
class OHLCVData:
    """Data OHLCV untuk semua timeframe."""
    ticker: str
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    data_age: DataAgeClassification
    days_available: int


# ---------------------------------------------------------------------------
# Dataclass - Analisis Pasar
# ---------------------------------------------------------------------------

@dataclass
class MarketRegimeResult:
    """Hasil deteksi regime pasar."""
    regime: MarketRegime
    active_indicators: List[str]
    description: str
    adx_value: float
    plus_di: float
    minus_di: float


@dataclass
class DimensionScore:
    """Skor untuk satu dimensi sinyal."""
    dimension: int
    name: str
    score: float
    max_score: float
    direction: SignalDirection
    triggered_indicators: List[str]
    weight: float


@dataclass
class SignalResult:
    """Hasil kalkulasi sinyal untuk satu timeframe."""
    dimension_scores: List[DimensionScore]
    total_score: float
    signal_strength: SignalStrength
    direction: SignalDirection
    timeframe: str
    alignment_score: float


# ---------------------------------------------------------------------------
# Dataclass - Divergensi & Support/Resistance
# ---------------------------------------------------------------------------

@dataclass
class DivergenceResult:
    """Hasil deteksi divergensi harga vs indikator."""
    divergence_type: DivergenceType
    strength: float
    indicators_confirming: List[str]
    price_pivot_indices: List[int]
    indicator_pivot_indices: List[int]
    price_pivot_values: List[float]
    indicator_pivot_values: List[float]


@dataclass
class SupportResistanceLevel:
    """Satu level support atau resistance."""
    price: float
    level_type: SRLevelType
    source: SRLevelSource
    source_detail: str
    touches: int
    distance_pct: float
    distance_atr: float


@dataclass
class SRResult:
    """Kumpulan level support/resistance beserta level terdekat."""
    all_levels: List[SupportResistanceLevel]
    nearest_support: Optional[SupportResistanceLevel]
    nearest_resistance: Optional[SupportResistanceLevel]
    current_price: float
    atr: float


# ---------------------------------------------------------------------------
# Dataclass - Manajemen Risiko & Rekomendasi
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Hasil kalkulasi risiko: entry, SL, TP, dan position sizing."""
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    sl_distance: float
    sl_distance_pct: float
    rr_ratio_tp1: Optional[float]
    rr_ratio_tp2: Optional[float]
    rr_ratio_tp3: Optional[float]
    position_size_lots: Optional[int]
    capital_at_risk: Optional[float]
    sl_method: str


@dataclass
class RecommendationCard:
    """Kartu rekomendasi untuk satu horizon investasi."""
    horizon: str
    horizon_detail: str
    signal: SignalDirection
    signal_strength: SignalStrength
    confidence_pct: float
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr_ratio: Optional[float]
    holding_duration: str
    explanation: List[str]
    timeframe_alignment: bool
    based_on_timeframe: str


@dataclass
class EmitenInfo:
    """Informasi dasar emiten / perusahaan tercatat."""
    ticker: str
    name: str
    listing_date: str
    sector: str
    days_listed: int
    ipo_price: Optional[float]
    current_price: Optional[float]
    price_change_vs_ipo_pct: Optional[float]


@dataclass
class AnalysisResult:
    """Hasil analisis lengkap untuk satu ticker."""
    ticker: str
    company_name: str
    current_price: float
    price_change_pct: float
    data_age: DataAgeClassification
    days_available: int
    market_regime: MarketRegimeResult
    signal_daily: SignalResult
    signal_weekly: SignalResult
    signal_monthly: SignalResult
    divergences: List[DivergenceResult]
    sr_result: SRResult
    risk_result: RiskResult
    recommendations: List[RecommendationCard]
    combined_narrative: str
    analysis_timestamp: str
