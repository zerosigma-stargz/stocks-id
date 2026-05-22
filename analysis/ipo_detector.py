"""
analysis/ipo_detector.py — Penanganan Saham IPO

Mendeteksi saham baru listing dan mengatur konfigurasi indikator
berdasarkan usia data (DataAgeClassification).
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from models import DataAgeClassification, EmitenInfo


# ---------------------------------------------------------------------------
# Konfigurasi Indikator per DataAgeClassification
# ---------------------------------------------------------------------------

# Tabel konfigurasi indikator berdasarkan klasifikasi usia data.
# Setiap key adalah nama indikator, value adalah dict {DataAgeClassification: bool}.
_IPO_CONFIG_TABLE: dict[str, dict[DataAgeClassification, bool]] = {
    'RSI':         {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: True,  DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'MACD':        {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: True,  DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'MA20':        {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'MA50':        {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'MA200':       {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: False, DataAgeClassification.FULL: True},
    'BB':          {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'Fibonacci':   {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'ADX':         {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'OBV':         {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'MFI':         {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
    'Candlestick': {DataAgeClassification.IPO_NEW: False, DataAgeClassification.IPO_PARTIAL: False, DataAgeClassification.STANDARD: True,  DataAgeClassification.FULL: True},
}


def get_ipo_config(data_age: DataAgeClassification) -> dict[str, bool]:
    """
    Mengembalikan dict indikator aktif berdasarkan klasifikasi usia data.

    Parameters
    ----------
    data_age : DataAgeClassification
        Klasifikasi usia data saham (IPO_NEW, IPO_PARTIAL, STANDARD, FULL).

    Returns
    -------
    dict[str, bool]
        Mapping nama indikator ke True/False.
        Keys: 'RSI', 'MACD', 'MA20', 'MA50', 'MA200', 'BB',
              'Fibonacci', 'ADX', 'OBV', 'MFI', 'Candlestick'
    """
    return {indicator: config[data_age] for indicator, config in _IPO_CONFIG_TABLE.items()}


# ---------------------------------------------------------------------------
# Filter Emiten Baru Listing
# ---------------------------------------------------------------------------

def _parse_listing_date(date_val) -> Optional[datetime]:
    """Parse berbagai format tanggal listing menjadi datetime object."""
    if date_val is None or (isinstance(date_val, float) and pd.isna(date_val)):
        return None
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, pd.Timestamp):
        return date_val.to_pydatetime()
    # Coba parse string
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(str(date_val).strip(), fmt)
        except ValueError:
            continue
    return None


def get_new_listings(emiten_df: pd.DataFrame, days: int = 7) -> list[EmitenInfo]:
    """
    Filter emiten yang listing dalam `days` hari terakhir.

    Parameters
    ----------
    emiten_df : pd.DataFrame
        DataFrame emiten dengan kolom: ticker, name, listing_date, sector.
    days : int, optional
        Jumlah hari ke belakang dari hari ini (default 7).

    Returns
    -------
    list[EmitenInfo]
        Daftar EmitenInfo untuk emiten yang listing dalam rentang waktu tersebut.
    """
    if emiten_df.empty:
        return []

    cutoff_date = datetime.now() - timedelta(days=days)
    result: list[EmitenInfo] = []

    for _, row in emiten_df.iterrows():
        listing_dt = _parse_listing_date(row.get('listing_date'))
        if listing_dt is None:
            continue

        if listing_dt >= cutoff_date:
            days_listed = (datetime.now() - listing_dt).days
            result.append(
                EmitenInfo(
                    ticker=str(row.get('ticker', '')),
                    name=str(row.get('name', '')),
                    listing_date=str(row.get('listing_date', '')),
                    sector=str(row.get('sector', '')),
                    days_listed=days_listed,
                    ipo_price=None,
                    current_price=None,
                    price_change_vs_ipo_pct=None,
                )
            )

    return result


# ---------------------------------------------------------------------------
# IPO Radar — Tabel Saham Baru 30 Hari
# ---------------------------------------------------------------------------

def get_ipo_radar(emiten_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Filter emiten yang listing dalam `days` hari terakhir dan tambahkan
    kolom days_listed serta price_change.

    Parameters
    ----------
    emiten_df : pd.DataFrame
        DataFrame emiten dengan kolom: ticker, name, listing_date, sector.
    days : int, optional
        Jumlah hari ke belakang dari hari ini (default 30).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom:
        ticker, name, listing_date, sector, days_listed, price_change
        Diurutkan dari yang paling baru listing (days_listed terkecil).
    """
    if emiten_df.empty:
        return pd.DataFrame(columns=['ticker', 'name', 'listing_date', 'sector', 'days_listed', 'price_change'])

    cutoff_date = datetime.now() - timedelta(days=days)
    rows = []

    for _, row in emiten_df.iterrows():
        listing_dt = _parse_listing_date(row.get('listing_date'))
        if listing_dt is None:
            continue

        if listing_dt >= cutoff_date:
            days_listed = (datetime.now() - listing_dt).days
            rows.append({
                'ticker': str(row.get('ticker', '')),
                'name': str(row.get('name', '')),
                'listing_date': str(row.get('listing_date', '')),
                'sector': str(row.get('sector', '')),
                'days_listed': int(days_listed),
                # price_change adalah placeholder — harga live tidak tersedia di sini
                'price_change': None,
            })

    if not rows:
        return pd.DataFrame(columns=['ticker', 'name', 'listing_date', 'sector', 'days_listed', 'price_change'])

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values('days_listed', ascending=True).reset_index(drop=True)
    return result_df
