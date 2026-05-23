"""
data/fetcher.py — Data Engine
Pengambilan data OHLCV dari yfinance untuk analisis teknikal saham BEI.
Kompatibel dengan yfinance 0.2.x dan 1.x.
"""

import logging
import os

import pandas as pd
import yfinance as yf
import streamlit as st

from models import DataAgeClassification

logger = logging.getLogger(__name__)

# Pastikan direktori cache tersedia
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Deteksi versi yfinance untuk penyesuaian API
try:
    _YF_VERSION = tuple(int(x) for x in yf.__version__.split(".")[:2])
except Exception:
    _YF_VERSION = (0, 2)

_YF_V1 = _YF_VERSION[0] >= 1  # True jika yfinance >= 1.0


# ---------------------------------------------------------------------------
# Helper internal
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratakan MultiIndex kolom yfinance menjadi kolom tunggal.
    Menangani berbagai format output yfinance 0.2.x dan 1.x.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    df.columns = df.columns.get_level_values(0)
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi nama kolom ke format standar: Open, High, Low, Close, Volume.
    Menangani variasi kapitalisasi dari berbagai versi yfinance.
    """
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    for standard in ['Open', 'High', 'Low', 'Close', 'Volume']:
        key = standard.lower()
        if key in col_lower and col_lower[key] != standard:
            rename_map[col_lower[key]] = standard

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom OHLCV tidak lengkap, kolom hilang: {missing}")

    return df[required]


def _download_with_fallback(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """
    Download data dengan strategi fallback untuk kompatibilitas yfinance 0.2.x dan 1.x.
    Mencoba beberapa kombinasi parameter secara berurutan.
    """
    errors = []

    # --- Strategi 1: yfinance 1.x — multi_level_index=False (kolom flat langsung) ---
    if _YF_V1:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if df is not None and not df.empty:
                logger.debug(f"[yf1.x flat] Berhasil download {ticker} {interval}")
                return df
        except Exception as e:
            errors.append(f"yf1.x flat: {e}")

    # --- Strategi 2: parameter standar (kompatibel 0.2.x dan 1.x) ---
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is not None and not df.empty:
            logger.debug(f"[yf standard] Berhasil download {ticker} {interval}")
            return df
    except Exception as e:
        errors.append(f"standard: {e}")

    # --- Strategi 3: via yf.Ticker().history() — lebih stabil di 1.x ---
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
        )
        if df is not None and not df.empty:
            logger.debug(f"[yf.Ticker.history] Berhasil download {ticker} {interval}")
            return df
    except Exception as e:
        errors.append(f"Ticker.history: {e}")

    logger.warning(
        f"Semua strategi download gagal untuk {ticker} {interval}: "
        + " | ".join(errors)
    )
    raise ValueError(
        f"Gagal mengambil data '{ticker}' (interval={interval}) dari yfinance. "
        f"Detail: {' | '.join(errors)}"
    )


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
    Kompatibel dengan yfinance 0.2.x dan 1.x.

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
    df = _download_with_fallback(ticker, start_date, end_date, interval)

    if df is None or df.empty:
        raise ValueError(
            f"Data OHLCV untuk '{ticker}' (interval={interval}) tidak tersedia "
            f"pada rentang {start_date} s/d {end_date}. "
            "Pastikan kode saham benar dan rentang tanggal valid."
        )

    df = _flatten_columns(df)
    df = _normalize_ohlcv(df)
    df = df.dropna(how='all')

    if df.empty:
        raise ValueError(
            f"Data OHLCV untuk '{ticker}' kosong setelah pembersihan. "
            "Coba perluas rentang tanggal."
        )

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
    Kompatibel dengan yfinance 0.2.x dan 1.x.

    Parameters
    ----------
    ticker : Kode saham, mis. 'BBCA.JK' atau '^JKSE'

    Returns
    -------
    dict dengan kunci:
        - 'name'          : Nama perusahaan
        - 'sector'        : Sektor industri
        - 'current_price' : Harga terakhir (float atau None)
    """
    info = {}
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info or {}
    except Exception as exc:
        logger.warning(f"Gagal mengambil .info untuk '{ticker}': {exc}")

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
    )

    if current_price is None:
        try:
            fast = yf.Ticker(ticker).fast_info
            current_price = getattr(fast, 'last_price', None)
        except Exception:
            pass

    name: str = (
        info.get("longName")
        or info.get("shortName")
        or ticker
    )

    sector: str = info.get("sector") or "N/A"

    return {
        "name": name,
        "sector": sector,
        "current_price": current_price,
    }
