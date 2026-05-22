"""
tests/test_data_engine.py — Unit tests untuk Data Engine

Memverifikasi:
- classify_data_age() mengembalikan enum yang benar untuk berbagai jumlah baris
- is_cache_valid() bekerja dengan benar berdasarkan isi last_sync.txt
- detect_new_listings() mendeteksi emiten baru dengan benar
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Tambahkan root proyek ke sys.path agar import berjalan dari direktori mana pun
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetcher import classify_data_age
from data.emiten_sync import is_cache_valid, detect_new_listings, LAST_SYNC_TXT, CACHE_TTL_DAYS
from models import DataAgeClassification, EmitenInfo


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(n_rows: int) -> pd.DataFrame:
    """Buat DataFrame OHLCV dummy dengan n_rows baris."""
    dates = pd.date_range(end=datetime.today(), periods=n_rows, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0] * n_rows,
            "High": [105.0] * n_rows,
            "Low": [95.0] * n_rows,
            "Close": [102.0] * n_rows,
            "Volume": [1_000_000] * n_rows,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests: classify_data_age()
# ---------------------------------------------------------------------------

class TestClassifyDataAge:
    """Verifikasi klasifikasi usia data berdasarkan jumlah baris DataFrame harian."""

    def test_ipo_new_5_rows(self):
        """5 baris → kurang dari 14 hari → IPO_NEW."""
        df = _make_df(5)
        result = classify_data_age(df)
        assert result == DataAgeClassification.IPO_NEW

    def test_ipo_new_boundary_13_rows(self):
        """13 baris → tepat di bawah batas 14 → IPO_NEW."""
        df = _make_df(13)
        result = classify_data_age(df)
        assert result == DataAgeClassification.IPO_NEW

    def test_ipo_partial_14_rows(self):
        """14 baris → tepat di batas bawah IPO_PARTIAL."""
        df = _make_df(14)
        result = classify_data_age(df)
        assert result == DataAgeClassification.IPO_PARTIAL

    def test_ipo_partial_20_rows(self):
        """20 baris → dalam rentang 14–60 → IPO_PARTIAL."""
        df = _make_df(20)
        result = classify_data_age(df)
        assert result == DataAgeClassification.IPO_PARTIAL

    def test_ipo_partial_boundary_59_rows(self):
        """59 baris → tepat di bawah batas 60 → IPO_PARTIAL."""
        df = _make_df(59)
        result = classify_data_age(df)
        assert result == DataAgeClassification.IPO_PARTIAL

    def test_standard_60_rows(self):
        """60 baris → tepat di batas bawah STANDARD."""
        df = _make_df(60)
        result = classify_data_age(df)
        assert result == DataAgeClassification.STANDARD

    def test_standard_100_rows(self):
        """100 baris → dalam rentang 60–365 → STANDARD."""
        df = _make_df(100)
        result = classify_data_age(df)
        assert result == DataAgeClassification.STANDARD

    def test_standard_boundary_365_rows(self):
        """365 baris → tepat di batas atas STANDARD (inklusif)."""
        df = _make_df(365)
        result = classify_data_age(df)
        assert result == DataAgeClassification.STANDARD

    def test_full_366_rows(self):
        """366 baris → tepat di atas batas 365 → FULL."""
        df = _make_df(366)
        result = classify_data_age(df)
        assert result == DataAgeClassification.FULL

    def test_full_400_rows(self):
        """400 baris → lebih dari 365 hari → FULL."""
        df = _make_df(400)
        result = classify_data_age(df)
        assert result == DataAgeClassification.FULL

    def test_full_large_dataset(self):
        """1000 baris → dataset besar → FULL."""
        df = _make_df(1000)
        result = classify_data_age(df)
        assert result == DataAgeClassification.FULL

    def test_empty_dataframe(self):
        """DataFrame kosong (0 baris) → IPO_NEW."""
        df = _make_df(0)
        result = classify_data_age(df)
        assert result == DataAgeClassification.IPO_NEW


# ---------------------------------------------------------------------------
# Tests: is_cache_valid()
# ---------------------------------------------------------------------------

class TestIsCacheValid:
    """Verifikasi logika validasi cache berdasarkan isi last_sync.txt."""

    def test_cache_valid_fresh_file(self, tmp_path):
        """File last_sync.txt baru (1 jam lalu) → cache valid."""
        sync_file = tmp_path / "last_sync.txt"
        sync_file.write_text(datetime.now().isoformat(), encoding="utf-8")

        with patch("data.emiten_sync.LAST_SYNC_TXT", sync_file):
            assert is_cache_valid() is True

    def test_cache_invalid_old_file(self, tmp_path):
        """File last_sync.txt berusia 8 hari → cache tidak valid."""
        sync_file = tmp_path / "last_sync.txt"
        old_time = datetime.now() - timedelta(days=8)
        sync_file.write_text(old_time.isoformat(), encoding="utf-8")

        with patch("data.emiten_sync.LAST_SYNC_TXT", sync_file):
            assert is_cache_valid() is False

    def test_cache_invalid_missing_file(self, tmp_path):
        """File last_sync.txt tidak ada → cache tidak valid."""
        missing_file = tmp_path / "last_sync.txt"
        # Pastikan file tidak ada
        assert not missing_file.exists()

        with patch("data.emiten_sync.LAST_SYNC_TXT", missing_file):
            assert is_cache_valid() is False

    def test_cache_invalid_corrupt_file(self, tmp_path):
        """File last_sync.txt berisi teks tidak valid → cache tidak valid."""
        sync_file = tmp_path / "last_sync.txt"
        sync_file.write_text("bukan-tanggal-valid", encoding="utf-8")

        with patch("data.emiten_sync.LAST_SYNC_TXT", sync_file):
            assert is_cache_valid() is False

    def test_cache_valid_exactly_at_ttl_minus_one(self, tmp_path):
        """File berusia tepat 6 hari 23 jam → masih valid."""
        sync_file = tmp_path / "last_sync.txt"
        almost_expired = datetime.now() - timedelta(days=CACHE_TTL_DAYS - 1)
        sync_file.write_text(almost_expired.isoformat(), encoding="utf-8")

        with patch("data.emiten_sync.LAST_SYNC_TXT", sync_file):
            assert is_cache_valid() is True

    def test_cache_invalid_exactly_at_ttl(self, tmp_path):
        """File berusia tepat 7 hari → sudah kedaluwarsa."""
        sync_file = tmp_path / "last_sync.txt"
        expired = datetime.now() - timedelta(days=CACHE_TTL_DAYS)
        sync_file.write_text(expired.isoformat(), encoding="utf-8")

        with patch("data.emiten_sync.LAST_SYNC_TXT", sync_file):
            assert is_cache_valid() is False


# ---------------------------------------------------------------------------
# Tests: detect_new_listings()
# ---------------------------------------------------------------------------

class TestDetectNewListings:
    """Verifikasi deteksi emiten baru listing."""

    def _make_emiten_df(self, rows: list) -> pd.DataFrame:
        """Buat DataFrame emiten dari list of dict."""
        return pd.DataFrame(rows, columns=["ticker", "name", "listing_date", "sector"])

    def test_detects_ticker_not_in_cache(self):
        """Ticker baru (tidak ada di cache) dengan listing_date baru → terdeteksi."""
        today = datetime.now().strftime("%Y-%m-%d")
        current = self._make_emiten_df([
            {"ticker": "BBCA.JK", "name": "BCA", "listing_date": "2000-05-31", "sector": "Keuangan"},
            {"ticker": "NEWX.JK", "name": "New Emiten", "listing_date": today, "sector": "Teknologi"},
        ])
        cached = self._make_emiten_df([
            {"ticker": "BBCA.JK", "name": "BCA", "listing_date": "2000-05-31", "sector": "Keuangan"},
        ])

        result = detect_new_listings(current, cached, days=30)
        tickers = [e.ticker for e in result]
        assert "NEWX.JK" in tickers

    def test_does_not_detect_old_ticker_in_cache(self):
        """Ticker lama yang sudah ada di cache → tidak terdeteksi sebagai baru."""
        current = self._make_emiten_df([
            {"ticker": "BBCA.JK", "name": "BCA", "listing_date": "2000-05-31", "sector": "Keuangan"},
        ])
        cached = self._make_emiten_df([
            {"ticker": "BBCA.JK", "name": "BCA", "listing_date": "2000-05-31", "sector": "Keuangan"},
        ])

        result = detect_new_listings(current, cached, days=30)
        tickers = [e.ticker for e in result]
        assert "BBCA.JK" not in tickers

    def test_detects_recent_listing_date(self):
        """Emiten dengan listing_date 10 hari lalu → terdeteksi dalam window 30 hari."""
        recent_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        current = self._make_emiten_df([
            {"ticker": "RCNT.JK", "name": "Recent IPO", "listing_date": recent_date, "sector": "Industri"},
        ])
        cached = self._make_emiten_df([
            {"ticker": "RCNT.JK", "name": "Recent IPO", "listing_date": recent_date, "sector": "Industri"},
        ])

        result = detect_new_listings(current, cached, days=30)
        tickers = [e.ticker for e in result]
        assert "RCNT.JK" in tickers

    def test_does_not_detect_old_listing_date(self):
        """Emiten dengan listing_date 60 hari lalu → tidak terdeteksi dalam window 30 hari."""
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        current = self._make_emiten_df([
            {"ticker": "OLDE.JK", "name": "Old IPO", "listing_date": old_date, "sector": "Industri"},
        ])
        cached = self._make_emiten_df([
            {"ticker": "OLDE.JK", "name": "Old IPO", "listing_date": old_date, "sector": "Industri"},
        ])

        result = detect_new_listings(current, cached, days=30)
        tickers = [e.ticker for e in result]
        assert "OLDE.JK" not in tickers

    def test_empty_current_df_returns_empty(self):
        """current_df kosong → kembalikan list kosong."""
        current = self._make_emiten_df([])
        cached = self._make_emiten_df([
            {"ticker": "BBCA.JK", "name": "BCA", "listing_date": "2000-05-31", "sector": "Keuangan"},
        ])

        result = detect_new_listings(current, cached, days=30)
        assert result == []

    def test_none_cached_df_uses_listing_date_only(self):
        """cached_df=None → hanya filter berdasarkan listing_date."""
        recent_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        old_date = "2000-01-01"
        current = self._make_emiten_df([
            {"ticker": "NEWX.JK", "name": "New", "listing_date": recent_date, "sector": "Teknologi"},
            {"ticker": "OLDE.JK", "name": "Old", "listing_date": old_date, "sector": "Keuangan"},
        ])

        result = detect_new_listings(current, None, days=30)
        tickers = [e.ticker for e in result]
        assert "NEWX.JK" in tickers
        assert "OLDE.JK" not in tickers

    def test_returns_emiten_info_objects(self):
        """Hasil adalah list EmitenInfo dengan field yang benar."""
        today = datetime.now().strftime("%Y-%m-%d")
        current = self._make_emiten_df([
            {"ticker": "NEWX.JK", "name": "New Emiten", "listing_date": today, "sector": "Teknologi"},
        ])
        cached = self._make_emiten_df([])

        result = detect_new_listings(current, cached, days=30)
        assert len(result) == 1
        emiten = result[0]
        assert isinstance(emiten, EmitenInfo)
        assert emiten.ticker == "NEWX.JK"
        assert emiten.name == "New Emiten"
        assert emiten.sector == "Teknologi"
        assert emiten.days_listed >= 0

    def test_sorted_by_listing_date_descending(self):
        """Hasil diurutkan berdasarkan listing_date terbaru terlebih dahulu."""
        date1 = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        date2 = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        current = self._make_emiten_df([
            {"ticker": "OLDR.JK", "name": "Older", "listing_date": date2, "sector": "Industri"},
            {"ticker": "NEWR.JK", "name": "Newer", "listing_date": date1, "sector": "Teknologi"},
        ])
        cached = self._make_emiten_df([])

        result = detect_new_listings(current, cached, days=30)
        assert len(result) == 2
        assert result[0].listing_date >= result[1].listing_date
