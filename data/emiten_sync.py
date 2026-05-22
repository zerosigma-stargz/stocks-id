"""
data/emiten_sync.py — Sinkronisasi Emiten BEI

Mengelola cache daftar emiten dari IDX (Bursa Efek Indonesia).
Menyediakan fungsi untuk memuat, menyinkronkan, dan mendeteksi emiten baru.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from models import EmitenInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

CACHE_DIR = Path("data/cache")
EMITEN_CSV = CACHE_DIR / "emiten_list.csv"
LAST_SYNC_TXT = CACHE_DIR / "last_sync.txt"
CACHE_TTL_DAYS = 7

IDX_URL = "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/"
IDX_API_URL = "https://www.idx.co.id/primary/StockData/GetSecuritiesStock"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.idx.co.id/",
}

# Fallback minimal jika semua sumber gagal
FALLBACK_EMITEN = [
    {"ticker": "BBCA.JK", "name": "Bank Central Asia Tbk", "listing_date": "2000-05-31", "sector": "Keuangan"},
    {"ticker": "TLKM.JK", "name": "Telkom Indonesia (Persero) Tbk", "listing_date": "1995-11-14", "sector": "Infrastruktur"},
    {"ticker": "BBRI.JK", "name": "Bank Rakyat Indonesia (Persero) Tbk", "listing_date": "2003-11-10", "sector": "Keuangan"},
    {"ticker": "BMRI.JK", "name": "Bank Mandiri (Persero) Tbk", "listing_date": "2003-07-14", "sector": "Keuangan"},
    {"ticker": "ASII.JK", "name": "Astra International Tbk", "listing_date": "1990-04-04", "sector": "Industri"},
]


# ---------------------------------------------------------------------------
# 4.1 is_cache_valid()
# ---------------------------------------------------------------------------

def is_cache_valid() -> bool:
    """
    Periksa apakah cache daftar emiten masih valid (< 7 hari).

    Returns:
        True jika last_sync.txt ada dan usianya < CACHE_TTL_DAYS hari,
        False jika file tidak ada, tidak bisa dibaca, atau sudah kedaluwarsa.
    """
    if not LAST_SYNC_TXT.exists():
        return False

    try:
        raw = LAST_SYNC_TXT.read_text(encoding="utf-8").strip()
        last_sync = datetime.fromisoformat(raw)
        age = datetime.now() - last_sync
        return age < timedelta(days=CACHE_TTL_DAYS)
    except (ValueError, OSError) as exc:
        logger.warning("Gagal membaca last_sync.txt: %s", exc)
        return False


# ---------------------------------------------------------------------------
# 4.2 sync_emiten_from_idx()
# ---------------------------------------------------------------------------

def _try_idx_api() -> Optional[pd.DataFrame]:
    """
    Coba ambil daftar emiten dari IDX JSON API (endpoint tidak resmi).
    Mengembalikan DataFrame atau None jika gagal.
    """
    try:
        params = {
            "start": 0,
            "length": 9999,
            "orderBy": "KodeEmiten",
            "orderType": "asc",
        }
        resp = requests.get(
            IDX_API_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        records = data.get("data", []) or data.get("Data", [])
        if not records:
            return None

        rows = []
        for item in records:
            ticker_raw = item.get("KodeEmiten", "") or item.get("StockCode", "")
            if not ticker_raw:
                continue
            ticker = ticker_raw.strip().upper() + ".JK"
            name = (item.get("NamaEmiten", "") or item.get("StockName", "")).strip()
            listing_date = (item.get("TanggalPencatatan", "") or item.get("ListingDate", "")).strip()
            sector = (item.get("Sektor", "") or item.get("Sector", "")).strip()

            # Normalisasi tanggal ke YYYY-MM-DD
            listing_date = _normalize_date(listing_date)

            rows.append({
                "ticker": ticker,
                "name": name,
                "listing_date": listing_date,
                "sector": sector,
            })

        if not rows:
            return None

        return pd.DataFrame(rows, columns=["ticker", "name", "listing_date", "sector"])

    except Exception as exc:
        logger.warning("IDX API gagal: %s", exc)
        return None


def _try_idx_html() -> Optional[pd.DataFrame]:
    """
    Coba scrape halaman HTML IDX menggunakan BeautifulSoup.
    Mengembalikan DataFrame atau None jika gagal.
    """
    try:
        resp = requests.get(IDX_URL, headers=REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Cari tabel dengan header yang relevan
        tables = soup.find_all("table")
        for table in tables:
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            # Cari tabel yang punya kolom kode/ticker
            if any(h in ("kode", "code", "ticker", "kode emiten") for h in headers):
                rows = []
                for tr in table.find_all("tr")[1:]:  # skip header row
                    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cells) >= 2:
                        ticker = cells[0].strip().upper() + ".JK"
                        name = cells[1].strip() if len(cells) > 1 else ""
                        listing_date = cells[2].strip() if len(cells) > 2 else ""
                        sector = cells[3].strip() if len(cells) > 3 else ""
                        listing_date = _normalize_date(listing_date)
                        rows.append({
                            "ticker": ticker,
                            "name": name,
                            "listing_date": listing_date,
                            "sector": sector,
                        })

                if rows:
                    return pd.DataFrame(rows, columns=["ticker", "name", "listing_date", "sector"])

        logger.warning("Tidak ditemukan tabel emiten di halaman IDX HTML.")
        return None

    except Exception as exc:
        logger.warning("IDX HTML scraping gagal: %s", exc)
        return None


def _normalize_date(date_str: str) -> str:
    """
    Normalisasi berbagai format tanggal ke YYYY-MM-DD.
    Mengembalikan string kosong jika tidak bisa diparse.
    """
    if not date_str:
        return ""

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return date_str  # kembalikan apa adanya jika tidak bisa diparse


def sync_emiten_from_idx() -> pd.DataFrame:
    """
    Sinkronisasi daftar emiten dari IDX dan simpan ke cache.

    Urutan percobaan:
    1. IDX JSON API
    2. IDX HTML scraping dengan BeautifulSoup
    3. Fallback minimal (5 emiten besar)

    Returns:
        DataFrame dengan kolom: ticker, name, listing_date, sector
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Coba API JSON terlebih dahulu
    df = _try_idx_api()

    # Jika API gagal, coba HTML scraping
    if df is None or df.empty:
        logger.info("Mencoba scraping HTML IDX...")
        df = _try_idx_html()

    # Jika keduanya gagal, gunakan fallback minimal
    if df is None or df.empty:
        logger.warning("Semua sumber IDX gagal. Menggunakan fallback minimal.")
        df = pd.DataFrame(FALLBACK_EMITEN, columns=["ticker", "name", "listing_date", "sector"])

    # Simpan ke cache
    try:
        df.to_csv(EMITEN_CSV, index=False, encoding="utf-8")
        LAST_SYNC_TXT.write_text(datetime.now().isoformat(), encoding="utf-8")
        logger.info("Cache emiten berhasil disimpan: %d emiten", len(df))
    except OSError as exc:
        logger.error("Gagal menyimpan cache emiten: %s", exc)

    return df


# ---------------------------------------------------------------------------
# 4.3 load_emiten_list()
# ---------------------------------------------------------------------------

def load_emiten_list() -> pd.DataFrame:
    """
    Muat daftar emiten BEI.

    Logika:
    - Jika cache valid (< 7 hari): baca dari emiten_list.csv
    - Jika cache tidak valid: panggil sync_emiten_from_idx()
    - Jika sync gagal: gunakan cache lama + tampilkan st.warning
    - Jika tidak ada cache sama sekali: gunakan fallback minimal

    Returns:
        DataFrame dengan kolom: ticker, name, listing_date, sector
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Cache masih valid — baca langsung dari file
    if is_cache_valid() and EMITEN_CSV.exists():
        try:
            df = pd.read_csv(EMITEN_CSV, encoding="utf-8", dtype=str)
            df = df.fillna("")
            logger.info("Memuat emiten dari cache: %d emiten", len(df))
            return df
        except Exception as exc:
            logger.warning("Gagal membaca cache emiten: %s. Akan sinkronisasi ulang.", exc)

    # Cache tidak valid atau rusak — coba sinkronisasi
    try:
        df = sync_emiten_from_idx()
        return df
    except Exception as exc:
        logger.error("sync_emiten_from_idx() gagal: %s", exc)

        # 4.5 Fallback ke cache lama jika sync gagal
        if EMITEN_CSV.exists():
            try:
                df = pd.read_csv(EMITEN_CSV, encoding="utf-8", dtype=str)
                df = df.fillna("")
                logger.warning("Menggunakan cache lama (%d emiten) karena sync gagal.", len(df))
                _show_stale_cache_warning()
                return df
            except Exception as read_exc:
                logger.error("Gagal membaca cache lama: %s", read_exc)

        # Tidak ada cache sama sekali — gunakan fallback minimal
        logger.warning("Tidak ada cache. Menggunakan fallback minimal.")
        _show_stale_cache_warning()
        return pd.DataFrame(FALLBACK_EMITEN, columns=["ticker", "name", "listing_date", "sector"])


def _show_stale_cache_warning() -> None:
    """
    Tampilkan peringatan Streamlit bahwa daftar emiten mungkin tidak mutakhir.
    Menggunakan try/except agar modul tetap bisa diimpor di luar konteks Streamlit.
    """
    try:
        import streamlit as st
        st.warning(
            "⚠️ Gagal memperbarui daftar emiten dari IDX. "
            "Daftar emiten yang ditampilkan mungkin tidak mutakhir. "
            "Periksa koneksi internet Anda dan muat ulang aplikasi."
        )
    except Exception:
        pass  # Di luar konteks Streamlit — abaikan


# ---------------------------------------------------------------------------
# 4.4 detect_new_listings()
# ---------------------------------------------------------------------------

def detect_new_listings(
    current_df: pd.DataFrame,
    cached_df: pd.DataFrame,
    days: int = 30,
) -> List[EmitenInfo]:
    """
    Deteksi emiten yang baru listing dalam `days` hari terakhir.

    Membandingkan current_df dengan cached_df untuk menemukan emiten baru,
    lalu memfilter berdasarkan listing_date < `days` hari dari hari ini.

    Args:
        current_df: DataFrame emiten terbaru (dari IDX atau cache baru).
        cached_df:  DataFrame emiten sebelumnya (cache lama).
        days:       Jumlah hari ke belakang untuk filter listing baru (default 30).

    Returns:
        List EmitenInfo untuk emiten yang listing dalam `days` hari terakhir.
    """
    if current_df is None or current_df.empty:
        return []

    cutoff_date = datetime.now() - timedelta(days=days)

    # Tentukan ticker baru (ada di current tapi tidak di cached)
    if cached_df is not None and not cached_df.empty:
        cached_tickers = set(cached_df["ticker"].str.upper())
        new_tickers = set(current_df["ticker"].str.upper()) - cached_tickers
        # Gabungkan: emiten baru ATAU emiten yang listing dalam `days` hari
        mask_new_ticker = current_df["ticker"].str.upper().isin(new_tickers)
    else:
        mask_new_ticker = pd.Series([False] * len(current_df), index=current_df.index)

    # Filter berdasarkan listing_date
    def _is_recent(date_str: str) -> bool:
        if not date_str or not isinstance(date_str, str):
            return False
        try:
            listing_dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            return listing_dt >= cutoff_date
        except ValueError:
            return False

    mask_recent_date = current_df["listing_date"].apply(_is_recent)

    # Gabungkan kedua kondisi
    combined_mask = mask_new_ticker | mask_recent_date
    filtered = current_df[combined_mask].copy()

    result: List[EmitenInfo] = []
    today = datetime.now().date()

    for _, row in filtered.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        name = str(row.get("name", "")).strip()
        listing_date_str = str(row.get("listing_date", "")).strip()
        sector = str(row.get("sector", "")).strip()

        # Hitung days_listed
        days_listed = 0
        if listing_date_str:
            try:
                listing_dt = datetime.strptime(listing_date_str, "%Y-%m-%d").date()
                days_listed = (today - listing_dt).days
            except ValueError:
                days_listed = 0

        emiten = EmitenInfo(
            ticker=ticker,
            name=name,
            listing_date=listing_date_str,
            sector=sector,
            days_listed=days_listed,
            ipo_price=None,
            current_price=None,
            price_change_vs_ipo_pct=None,
        )
        result.append(emiten)

    # Urutkan berdasarkan listing_date terbaru
    result.sort(key=lambda e: e.listing_date, reverse=True)
    return result
