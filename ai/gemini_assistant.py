"""
ai/gemini_assistant.py — Gemini AI Assistant untuk StockMomentum ID

Menyediakan:
  - Koneksi ke Google Gemini API (gemini-2.5-flash)
  - Pembacaan konten dari URL berita
  - Pembangunan konteks analisis teknikal dari AnalysisResult
  - Fungsi chat dengan riwayat percakapan
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

_MODEL_NAME = 'gemini-2.5-flash'
_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\']+',
    re.IGNORECASE,
)
_FETCH_TIMEOUT = 10       # detik
_MAX_URL_CONTENT = 3000   # karakter maksimum konten URL yang dikirim ke Gemini
_MAX_HISTORY = 20         # maksimum pasang pesan dalam riwayat


# ---------------------------------------------------------------------------
# Inisialisasi Gemini
# ---------------------------------------------------------------------------

def init_gemini(api_key: str):
    """
    Inisialisasi klien Gemini dengan API key yang diberikan.

    Parameters
    ----------
    api_key : str
        Google Gemini API key.

    Returns
    -------
    GenerativeModel atau None jika gagal.
    """
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=_MODEL_NAME,
            system_instruction=_build_system_prompt(),
        )
        return model
    except ImportError:
        logger.error(
            "google-generativeai tidak terinstall. "
            "Jalankan: pip install google-generativeai"
        )
        return None
    except Exception as e:
        logger.error(f"Gagal menginisialisasi Gemini: {e}")
        return None


def _build_system_prompt() -> str:
    """Bangun system prompt untuk Gemini sebagai AI Assistant analisis saham BEI."""
    return """Kamu adalah AI Assistant analisis saham BEI (Bursa Efek Indonesia) bernama **StockMind**, 
yang terintegrasi dalam aplikasi StockMomentum ID.

## Peranmu
- Membantu pengguna memahami indikator teknikal (RSI, MACD, Bollinger Bands, ADX, dll.)
- Menginterpretasikan sinyal beli/jual/netral berdasarkan data analisis yang diberikan
- Membaca dan menginterpretasikan berita/artikel dari URL yang diberikan pengguna
- Menghubungkan berita fundamental dengan kondisi teknikal saham
- Memberikan edukasi tentang analisis teknikal dalam Bahasa Indonesia

## Panduan Respons
- Selalu gunakan **Bahasa Indonesia** yang jelas dan mudah dipahami
- Sertakan angka indikator yang relevan saat menjelaskan
- Bedakan antara analisis teknikal (dari data) dan interpretasi berita (dari URL)
- Selalu ingatkan bahwa ini adalah alat bantu, bukan saran investasi
- Gunakan format markdown untuk keterbacaan (bold, bullet points, dll.)
- Jika ada URL berita, ringkas isinya dan hubungkan dengan kondisi teknikal

## Batasan
- Tidak memberikan rekomendasi investasi yang pasti
- Tidak menjamin akurasi prediksi harga
- Selalu sarankan pengguna untuk melakukan riset mandiri (DYOR)"""


# ---------------------------------------------------------------------------
# Pembacaan Konten URL
# ---------------------------------------------------------------------------

def extract_urls(text: str) -> list[str]:
    """
    Ekstrak semua URL dari teks input pengguna.

    Parameters
    ----------
    text : str
        Teks input yang mungkin mengandung URL.

    Returns
    -------
    list[str]
        Daftar URL yang ditemukan.
    """
    return _URL_PATTERN.findall(text)


def fetch_url_content(url: str) -> str:
    """
    Ambil konten teks dari URL (berita/artikel).

    Parameters
    ----------
    url : str
        URL yang akan diambil kontennya.

    Returns
    -------
    str
        Konten teks yang sudah dibersihkan, atau pesan error.
    """
    try:
        # Validasi URL
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return f'[URL tidak valid: {url}]'

        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
        }

        resp = requests.get(url, headers=headers, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()

        # Parse HTML dengan BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Hapus elemen yang tidak relevan
            for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                             'aside', 'advertisement', 'ads']):
                tag.decompose()

            # Ambil teks dari elemen konten utama
            content_tags = soup.find_all(['article', 'main', 'div'], class_=re.compile(
                r'content|article|post|news|body|text', re.I
            ))

            if content_tags:
                text = ' '.join(tag.get_text(separator=' ', strip=True) for tag in content_tags[:3])
            else:
                text = soup.get_text(separator=' ', strip=True)

            # Bersihkan whitespace berlebih
            text = re.sub(r'\s+', ' ', text).strip()

            # Batasi panjang konten
            if len(text) > _MAX_URL_CONTENT:
                text = text[:_MAX_URL_CONTENT] + '...[konten dipotong]'

            return text if text else f'[Konten tidak dapat diekstrak dari: {url}]'

        except ImportError:
            # Fallback tanpa BeautifulSoup
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:_MAX_URL_CONTENT] if text else f'[Konten kosong: {url}]'

    except requests.exceptions.Timeout:
        return f'[Timeout saat mengakses: {url}]'
    except requests.exceptions.ConnectionError:
        return f'[Tidak dapat terhubung ke: {url}]'
    except requests.exceptions.HTTPError as e:
        return f'[HTTP Error {e.response.status_code} untuk: {url}]'
    except Exception as e:
        logger.warning(f"Gagal fetch URL {url}: {e}")
        return f'[Gagal membaca konten dari: {url}]'


# ---------------------------------------------------------------------------
# Pembangunan Konteks Analisis
# ---------------------------------------------------------------------------

def build_analysis_context(analysis_result=None, df_daily=None) -> str:
    """
    Bangun string konteks dari AnalysisResult dan DataFrame indikator
    untuk disertakan dalam prompt ke Gemini.

    Parameters
    ----------
    analysis_result : AnalysisResult, optional
        Hasil analisis lengkap dari engine.
    df_daily : pd.DataFrame, optional
        DataFrame harian dengan kolom indikator.

    Returns
    -------
    str
        String konteks yang siap disertakan dalam prompt.
    """
    if analysis_result is None:
        return ''

    try:
        from recommendation.risk_calculator import format_rupiah
        from models import SignalDirection

        r = analysis_result
        lines: list[str] = []

        # --- Info Saham ---
        lines.append(f'## Konteks Analisis: {r.company_name} ({r.ticker})')
        lines.append(f'- Harga Terakhir: {format_rupiah(r.current_price)}')
        lines.append(f'- Perubahan Harian: {r.price_change_pct:+.2f}%')
        lines.append(f'- Kondisi Pasar: {r.market_regime.regime.value}')
        lines.append(f'- Klasifikasi Data: {r.data_age.value} ({r.days_available} hari)')
        lines.append(f'- Waktu Analisis: {r.analysis_timestamp}')
        lines.append('')

        # --- Sinyal Multi-Timeframe ---
        lines.append('### Sinyal Teknikal')
        for sig, label in [
            (r.signal_daily,   'Harian (Jangka Pendek)'),
            (r.signal_weekly,  'Mingguan (Jangka Menengah)'),
            (r.signal_monthly, 'Bulanan (Jangka Panjang)'),
        ]:
            lines.append(
                f'- {label}: **{sig.direction.value}** '
                f'({sig.signal_strength.value}, skor {sig.total_score:.1f}/31)'
            )
        lines.append('')

        # --- Skor Dimensi (dari sinyal harian) ---
        lines.append('### Skor Dimensi (Harian)')
        for dim in r.signal_daily.dimension_scores:
            lines.append(
                f'- Dimensi {dim.dimension} ({dim.name}): '
                f'{dim.score:.1f}/{dim.max_score:.1f}'
            )
            if dim.triggered_indicators:
                for ind in dim.triggered_indicators[:3]:
                    lines.append(f'  • {ind}')
        lines.append('')

        # --- Nilai Indikator Terakhir ---
        if df_daily is not None and not df_daily.empty:
            last = df_daily.iloc[-1]
            lines.append('### Nilai Indikator Saat Ini')

            def _v(col: str, decimals: int = 2) -> str:
                import numpy as np
                import pandas as pd
                val = last.get(col, None)
                if val is None or (hasattr(val, '__float__') and pd.isna(float(val))):
                    return 'N/A'
                return f'{float(val):.{decimals}f}'

            def _v_prefix(prefix: str, decimals: int = 2) -> str:
                import pandas as pd
                for col in df_daily.columns:
                    if col.startswith(prefix + '_'):
                        val = last.get(col, None)
                        if val is not None:
                            try:
                                f = float(val)
                                if not pd.isna(f):
                                    return f'{f:.{decimals}f}'
                            except (TypeError, ValueError):
                                pass
                return 'N/A'

            indicator_vals = [
                ('RSI(14)',      _v('RSI_14')),
                ('MACD',         _v('MACD_12_26_9')),
                ('MACD Signal',  _v('MACDs_12_26_9')),
                ('MACD Hist',    _v('MACDh_12_26_9')),
                ('ADX(14)',      _v('ADX_14')),
                ('+DI',          _v('DMP_14')),
                ('-DI',          _v('DMN_14')),
                ('MA20',         _v('SMA_20', 0)),
                ('MA50',         _v('SMA_50', 0)),
                ('MA200',        _v('SMA_200', 0)),
                ('BB Upper',     _v_prefix('BBU', 0)),
                ('BB Middle',    _v_prefix('BBM', 0)),
                ('BB Lower',     _v_prefix('BBL', 0)),
                ('ATR(14)',      _v('ATRr_14', 0)),
                ('MFI(14)',      _v('MFI_14')),
                ('Stoch %K',     _v_prefix('STOCHk')),
                ('Williams %R',  _v('WILLR_14')),
                ('CCI(20)',      _v('CCI_20')),
            ]
            for name, val in indicator_vals:
                if val != 'N/A':
                    lines.append(f'- {name}: {val}')
            lines.append('')

        # --- Rekomendasi ---
        lines.append('### Rekomendasi')
        for card in r.recommendations:
            lines.append(
                f'- {card.horizon} ({card.horizon_detail}): '
                f'**{card.signal.value}** — Confidence {card.confidence_pct:.0f}%'
            )
            if card.signal != SignalDirection.NETRAL and card.entry_price > 0:
                lines.append(
                    f'  Entry: {format_rupiah(card.entry_price)} | '
                    f'SL: {format_rupiah(card.stop_loss)} | '
                    f'TP1: {format_rupiah(card.tp1)}'
                )
        lines.append('')

        # --- Divergensi ---
        if r.divergences:
            lines.append('### Divergensi Terdeteksi')
            for div in r.divergences[:3]:
                lines.append(
                    f'- {div.divergence_type.value} '
                    f'(kekuatan: {div.strength:.2f}, '
                    f'indikator: {", ".join(div.indicators_confirming)})'
                )
            lines.append('')

        # --- Narasi ---
        lines.append('### Narasi Interpretasi')
        lines.append(r.combined_narrative)

        return '\n'.join(lines)

    except Exception as e:
        logger.warning(f"Gagal membangun konteks analisis: {e}")
        return ''


# ---------------------------------------------------------------------------
# Fungsi Chat Utama
# ---------------------------------------------------------------------------

def chat_with_gemini(
    model,
    user_message: str,
    chat_history: list[dict],
    analysis_context: str = '',
) -> tuple[str, list[dict]]:
    """
    Kirim pesan ke Gemini dan kembalikan respons beserta riwayat yang diperbarui.

    Parameters
    ----------
    model : GenerativeModel
        Model Gemini yang sudah diinisialisasi.
    user_message : str
        Pesan dari pengguna.
    chat_history : list[dict]
        Riwayat percakapan: [{'role': 'user'/'model', 'parts': [str]}, ...]
    analysis_context : str
        Konteks analisis teknikal yang akan disertakan.

    Returns
    -------
    tuple[str, list[dict]]
        (respons_teks, riwayat_diperbarui)
    """
    try:
        # Ekstrak dan fetch URL dari pesan pengguna
        urls = extract_urls(user_message)
        url_contents: list[str] = []

        for url in urls[:3]:  # Maksimal 3 URL per pesan
            content = fetch_url_content(url)
            url_contents.append(f'\n**Konten dari {url}:**\n{content}')

        # Bangun pesan lengkap dengan konteks
        full_message_parts: list[str] = []

        # Sertakan konteks analisis hanya pada pesan pertama atau jika ada konteks baru
        if analysis_context and not chat_history:
            full_message_parts.append(
                f'[KONTEKS ANALISIS TEKNIKAL — gunakan sebagai referensi]\n'
                f'{analysis_context}\n'
                f'[AKHIR KONTEKS]\n'
            )

        full_message_parts.append(user_message)

        if url_contents:
            full_message_parts.append(
                '\n\n[KONTEN URL YANG DIMINTA PENGGUNA]\n' +
                '\n'.join(url_contents) +
                '\n[AKHIR KONTEN URL]'
            )

        full_message = '\n'.join(full_message_parts)

        # Batasi riwayat agar tidak terlalu panjang
        trimmed_history = chat_history[-(_MAX_HISTORY * 2):]

        # Mulai sesi chat dengan riwayat
        chat_session = model.start_chat(history=trimmed_history)
        response = chat_session.send_message(full_message)

        response_text = response.text

        # Perbarui riwayat (simpan pesan asli pengguna, bukan yang sudah diperluas)
        updated_history = trimmed_history + [
            {'role': 'user',  'parts': [user_message]},
            {'role': 'model', 'parts': [response_text]},
        ]

        return response_text, updated_history

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error saat chat dengan Gemini: {error_msg}")

        # Pesan error yang informatif
        if 'API_KEY_INVALID' in error_msg or 'api key' in error_msg.lower():
            return (
                '❌ **API Key tidak valid.** Pastikan Gemini API key yang dimasukkan benar. '
                'Dapatkan API key gratis di [Google AI Studio](https://aistudio.google.com/).',
                chat_history,
            )
        elif 'quota' in error_msg.lower() or 'rate' in error_msg.lower():
            return (
                '⚠️ **Batas kuota tercapai.** Coba lagi beberapa saat. '
                'Model gemini-2.5-flash memiliki kuota gratis yang cukup besar.',
                chat_history,
            )
        elif 'SAFETY' in error_msg:
            return (
                '⚠️ **Respons diblokir oleh filter keamanan Gemini.** '
                'Coba reformulasikan pertanyaan Anda.',
                chat_history,
            )
        else:
            return (
                f'❌ **Terjadi kesalahan:** {error_msg}\n\n'
                'Periksa koneksi internet dan coba lagi.',
                chat_history,
            )
