"""
analysis/indicators.py — Perhitungan Indikator Teknikal

Menghitung semua indikator teknikal menggunakan pandas-ta dan menerapkan
konfigurasi IPO berdasarkan DataAgeClassification.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from models import DataAgeClassification

# Import pandas_ta dengan graceful fallback
try:
    import pandas_ta as ta
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    _PANDAS_TA_AVAILABLE = False
    logging.warning(
        "pandas_ta tidak tersedia. Indikator teknikal tidak dapat dihitung. "
        "Jalankan: pip install pandas-ta"
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
    if not _PANDAS_TA_AVAILABLE:
        logger.error("pandas_ta tidak tersedia — tidak dapat menghitung indikator.")
        return df

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
    pandas-ta kadang menghasilkan kolom bertipe int yang tidak bisa menyimpan NaN.
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


def _calc_sma(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung SMA 20, 50, 200."""
    # SMA_20
    if ipo_config.get('MA20', False) and n >= 20:
        result = df.ta.sma(length=20)
        if result is not None:
            df['SMA_20'] = result
        else:
            df['SMA_20'] = np.nan
    else:
        df['SMA_20'] = np.nan

    # SMA_50
    if ipo_config.get('MA50', False) and n >= 50:
        result = df.ta.sma(length=50)
        if result is not None:
            df['SMA_50'] = result
        else:
            df['SMA_50'] = np.nan
    else:
        df['SMA_50'] = np.nan

    # SMA_200
    if ipo_config.get('MA200', False) and n >= 200:
        result = df.ta.sma(length=200)
        if result is not None:
            df['SMA_200'] = result
        else:
            df['SMA_200'] = np.nan
    else:
        df['SMA_200'] = np.nan


def _calc_adx(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung ADX(14), +DI (DMP_14), -DI (DMN_14)."""
    adx_cols = _INDICATOR_COLUMNS['ADX']

    if ipo_config.get('ADX', False) and n >= 28:
        try:
            result = df.ta.adx(length=14)
            if result is not None and not result.empty:
                for col in result.columns:
                    df[col] = result[col]
                for col in adx_cols:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                _set_nan_columns(df, adx_cols)
        except Exception as e:
            logger.warning(f"Gagal menghitung ADX: {e}")
            _set_nan_columns(df, adx_cols)
    else:
        _set_nan_columns(df, adx_cols)


def _calc_rsi(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung RSI(14)."""
    if ipo_config.get('RSI', False) and n >= 14:
        try:
            result = df.ta.rsi(length=14)
            if result is not None:
                df['RSI_14'] = result
            else:
                df['RSI_14'] = np.nan
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
            result = df.ta.macd(fast=12, slow=26, signal=9)
            if result is not None and not result.empty:
                for col in result.columns:
                    df[col] = result[col]
                for col in macd_cols:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                _set_nan_columns(df, macd_cols)
        except Exception as e:
            logger.warning(f"Gagal menghitung MACD: {e}")
            _set_nan_columns(df, macd_cols)
    else:
        _set_nan_columns(df, macd_cols)


def _calc_roc(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung ROC(10) dan ROC(20)."""
    if standard_active and n >= 20:
        try:
            roc10 = df.ta.roc(length=10)
            roc20 = df.ta.roc(length=20)
            df['ROC_10'] = roc10 if roc10 is not None else np.nan
            df['ROC_20'] = roc20 if roc20 is not None else np.nan
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
            result = df.ta.willr(length=14)
            df['WILLR_14'] = result if result is not None else np.nan
        except Exception as e:
            logger.warning(f"Gagal menghitung Williams %R: {e}")
            df['WILLR_14'] = np.nan
    else:
        df['WILLR_14'] = np.nan


def _calc_cci(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung CCI(20)."""
    if standard_active and n >= 20:
        try:
            result = df.ta.cci(length=20)
            df['CCI_20'] = result if result is not None else np.nan
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
            result = df.ta.bbands(length=20, std=2)
            if result is not None and not result.empty:
                for col in result.columns:
                    df[col] = result[col]

                # Mapping fleksibel: pandas-ta bisa menghasilkan BBL_20_2.0 atau BBL_20_2
                _map_bb_columns(df)

                for col in bb_cols:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                _set_nan_columns(df, bb_cols)
        except Exception as e:
            logger.warning(f"Gagal menghitung Bollinger Bands: {e}")
            _set_nan_columns(df, bb_cols)
    else:
        _set_nan_columns(df, bb_cols)


def _map_bb_columns(df: pd.DataFrame) -> None:
    """
    Petakan kolom BB dari pandas-ta ke nama standar.

    pandas-ta bisa menghasilkan:
      - BBL_20_2.0 atau BBL_20_2
      - BBM_20_2.0 atau BBM_20_2
      - BBU_20_2.0 atau BBU_20_2
      - BBB_20_2.0 atau BBB_20_2
      - BBP_20_2.0 atau BBP_20_2
    """
    all_cols = list(df.columns)

    col_candidates = {
        'BBL_20_2.0': [c for c in all_cols if c.startswith('BBL_')],
        'BBM_20_2.0': [c for c in all_cols if c.startswith('BBM_')],
        'BBU_20_2.0': [c for c in all_cols if c.startswith('BBU_')],
        'BBB_20_2.0': [c for c in all_cols if c.startswith('BBB_')],
    }

    for target_col, candidates in col_candidates.items():
        if target_col not in df.columns and candidates:
            df[target_col] = df[candidates[0]]


def _calc_atr(df: pd.DataFrame, standard_active: bool, n: int) -> None:
    """Hitung ATR(14)."""
    if standard_active and n >= 14:
        try:
            result = df.ta.atr(length=14)
            df['ATRr_14'] = result if result is not None else np.nan
        except Exception as e:
            logger.warning(f"Gagal menghitung ATR: {e}")
            df['ATRr_14'] = np.nan
    else:
        df['ATRr_14'] = np.nan


def _calc_obv(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung OBV."""
    if ipo_config.get('OBV', False) and n >= 2:
        try:
            result = df.ta.obv()
            df['OBV'] = result if result is not None else np.nan
        except Exception as e:
            logger.warning(f"Gagal menghitung OBV: {e}")
            df['OBV'] = np.nan
    else:
        df['OBV'] = np.nan


def _calc_mfi(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung MFI(14)."""
    if ipo_config.get('MFI', False) and n >= 14:
        try:
            result = df.ta.mfi(length=14)
            df['MFI_14'] = result if result is not None else np.nan
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
            result = df.ta.stoch(k=14, d=3, smooth_k=3)
            if result is not None and not result.empty:
                for col in result.columns:
                    df[col] = result[col]
                for col in stoch_cols:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                _set_nan_columns(df, stoch_cols)
        except Exception as e:
            logger.warning(f"Gagal menghitung Stochastic: {e}")
            _set_nan_columns(df, stoch_cols)
    else:
        _set_nan_columns(df, stoch_cols)


def _calc_candlestick(df: pd.DataFrame, ipo_config: dict, n: int) -> None:
    """Hitung pola candlestick via pandas-ta."""
    cdl_cols = _INDICATOR_COLUMNS['Candlestick']

    if ipo_config.get('Candlestick', False) and n >= 3:
        patterns = {
            'CDL_HAMMER':       'hammer',
            'CDL_SHOOTING_STAR': 'shootingstar',
            'CDL_DOJI_10_0.1':  'doji',
            'CDL_ENGULFING':    'engulfing',
        }
        for col_name, pattern_name in patterns.items():
            try:
                result = df.ta.cdl_pattern(name=pattern_name)
                if result is not None and not result.empty:
                    # cdl_pattern returns a DataFrame; take the first column
                    first_col = result.columns[0]
                    df[col_name] = result[first_col]
                else:
                    df[col_name] = np.nan
            except Exception as e:
                logger.warning(f"Gagal menghitung pola candlestick '{pattern_name}': {e}")
                df[col_name] = np.nan
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
