"""
data/fetcher.py — Data Engine
Pengambilan data OHLCV dari yfinance untuk analisis teknikal saham BEI.
"""

import os
import pandas as pd
import yfinance as yf
import streamlit as st

from models import DataAgeClassification

# Pastikan direktori cache tersedia
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3.1  fetch_ohlcv — pengambilan data satu timeframe dengan cache Streamlit
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Mengambil data OHLCV dari yfinance untuk satu timeframe.

    Parameters
    ----------
    ticker     : Kode saham, mis. 'BBCA.JK' atau '^JKSE'
    start_date : Tanggal mulai dalam format 'YYYY-MM-DD'
    end_date   : Tanggal akhir dalam format 'YYYY-MM-DD'
    interval   : Interval data — '1d', '1wk', atau '1mo'

    Returns
    -------
    pd.DataFrame dengan kolom Open, High, Low, Close, Volume

    Raises
    ------
    ValueError  : Jika yfinance mengembalikan data kosong atau terjadi error
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Gagal mengambil data '{ticker}' (interval={interval}) dari yfinance: {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"Data OHLCV untuk '{ticker}' (interval={interval}) tidak tersedia "
            f"pada rentang {start_date} s/d {end_date}. "
            "Pastikan kode saham benar dan rentang tanggal valid."
        )

    # Ratakan MultiIndex kolom jika ada (yfinance kadang mengembalikan MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# ---------------------------------------------------------------------------
# 3.2  fetch_all_timeframes — ambil data untuk ketiga timeframe sekaligus
# ---------------------------------------------------------------------------

def fetch_all_timeframes(
    ticker: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """
    Mengambil data OHLCV untuk tiga timeframe: harian, mingguan, dan bulanan.

    Parameters
    ----------
    ticker     : Kode saham, mis. 'BBCA.JK' atau '^JKSE'
    start_date : Tanggal mulai dalam format 'YYYY-MM-DD'
    end_date   : Tanggal akhir dalam format 'YYYY-MM-DD'

    Returns
    -------
    dict dengan kunci '1d', '1wk', '1mo' dan nilai pd.DataFrame masing-masing

    Raises
    ------
    ValueError  : Diteruskan dari fetch_ohlcv jika data tidak tersedia
    """
    intervals = ["1d", "1wk", "1mo"]
    result: dict[str, pd.DataFrame] = {}

    for interval in intervals:
        result[interval] = fetch_ohlcv(ticker, start_date, end_date, interval)

    return result


# ---------------------------------------------------------------------------
# 3.3  classify_data_age — klasifikasi berdasarkan jumlah hari data harian
# ---------------------------------------------------------------------------

def classify_data_age(df_daily: pd.DataFrame) -> DataAgeClassification:
    """
    Mengklasifikasikan usia data berdasarkan jumlah baris pada DataFrame harian.

    Klasifikasi:
        < 14 hari   → IPO_NEW
        14–60 hari  → IPO_PARTIAL
        60–365 hari → STANDARD
        > 365 hari  → FULL

    Parameters
    ----------
    df_daily : DataFrame OHLCV dengan interval harian

    Returns
    -------
    DataAgeClassification
    """
    days = len(df_daily)

    if days < 14:
        return DataAgeClassification.IPO_NEW
    elif days < 60:
        return DataAgeClassification.IPO_PARTIAL
    elif days <= 365:
        return DataAgeClassification.STANDARD
    else:
        return DataAgeClassification.FULL


# ---------------------------------------------------------------------------
# 3.4  get_stock_info — nama perusahaan, sektor, dan harga terakhir
# ---------------------------------------------------------------------------

def get_stock_info(ticker: str) -> dict:
    """
    Mengambil informasi dasar emiten dari yfinance.

    Parameters
    ----------
    ticker : Kode saham, mis. 'BBCA.JK' atau '^JKSE'

    Returns
    -------
    dict dengan kunci:
        - 'name'          : Nama perusahaan (longName / shortName / ticker)
        - 'sector'        : Sektor industri (str atau 'N/A')
        - 'current_price' : Harga terakhir (float atau None)

    Raises
    ------
    ValueError : Jika yfinance gagal mengambil informasi ticker
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        raise ValueError(
            f"Gagal mengambil informasi untuk ticker '{ticker}': {exc}"
        ) from exc

    if not info:
        raise ValueError(
            f"Informasi untuk ticker '{ticker}' tidak tersedia. "
            "Pastikan kode saham benar."
        )

    name: str = (
        info.get("longName")
        or info.get("shortName")
        or ticker
    )

    sector: str = info.get("sector") or "N/A"

    current_price: float | None = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
    )

    return {
        "name": name,
        "sector": sector,
        "current_price": current_price,
    }
