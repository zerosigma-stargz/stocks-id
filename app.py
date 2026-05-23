"""
StockMomentum ID — Aplikasi Analisis Teknikal Saham BEI
Entry point utama aplikasi Streamlit.

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi halaman — HARUS dipanggil pertama kali
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='StockMomentum ID',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import modul internal
# ---------------------------------------------------------------------------

from models import (
    AnalysisResult,
    DataAgeClassification,
    SignalDirection,
    SignalStrength,
)
from recommendation.risk_calculator import format_rupiah
from recommendation.engine import all_signals_neutral

# ---------------------------------------------------------------------------
# Konstanta warna
# ---------------------------------------------------------------------------

_COLOR_BELI   = '#00C851'
_COLOR_JUAL   = '#FF4444'
_COLOR_NETRAL = '#9E9E9E'

_REGIME_COLORS = {
    'BULL_TREND': '#00C851',
    'BEAR_TREND': '#FF4444',
    'SIDEWAYS':   '#FFC107',
    'BREAKOUT':   '#2196F3',
}

_DATA_AGE_LABELS = {
    DataAgeClassification.IPO_NEW:     'IPO Baru',
    DataAgeClassification.IPO_PARTIAL: 'IPO Parsial',
    DataAgeClassification.STANDARD:    'Standar',
    DataAgeClassification.FULL:        'Lengkap',
}


# ===========================================================================
# SIDEBAR — Task 18.1, 18.2, 18.3, 18.4
# ===========================================================================

def render_sidebar() -> dict:
    """Render sidebar dan kembalikan parameter input pengguna."""
    with st.sidebar:
        st.markdown('**StockMomentum ID**')
        st.divider()

        # Task 18.2 — Autocomplete dari emiten_list
        emiten_df = _load_emiten_list_safe()
        ticker_options: list[str] = []
        if emiten_df is not None and not emiten_df.empty and 'ticker' in emiten_df.columns:
            ticker_options = emiten_df['ticker'].dropna().tolist()

        # Cek override dari IPO banner
        default_ticker = st.session_state.pop('ticker_override', '')

        ticker_input = st.text_input(
            'Ticker',
            value=default_ticker,
            placeholder='Contoh: BBCA, TLKM, ^JKSE',
            help='Masukkan kode saham BEI (tanpa .JK) atau ^JKSE untuk IHSG',
        ).strip().upper()

        if ticker_input and ticker_options:
            matches = [t for t in ticker_options if ticker_input in t and t != ticker_input][:5]
            if matches:
                st.caption(f'Saran: {", ".join(matches)}')

        today = date.today()
        col1, col2 = st.columns(2)
        start_date = col1.date_input('Dari', value=today - timedelta(days=730), max_value=today)
        end_date   = col2.date_input('Sampai', value=today, max_value=today)

        timeframe = st.selectbox('Timeframe Utama', ['Harian', 'Mingguan', 'Bulanan'])

        total_capital = st.number_input(
            'Modal (Rp) — opsional',
            min_value=0, value=0, step=1_000_000,
            help='Isi untuk menghitung ukuran posisi (position sizing)',
        )

        st.divider()
        analyze = st.button('Analisis', type='primary', use_container_width=True)

        _render_ipo_banner_sidebar(emiten_df)
        st.divider()

        # --- Gemini AI Assistant — API Key ---
        st.markdown('**Gemini API Key**')
        st.caption('Dapatkan API key gratis di [Google AI Studio](https://aistudio.google.com/apikey)')
        gemini_key = st.text_input(
            'Gemini API Key',
            value=st.session_state.get('gemini_api_key', ''),
            type='password',
            placeholder='AIzaSy...',
            help='API key dari Google AI Studio. Gratis, tidak perlu kartu kredit.',
            key='gemini_key_input',
            label_visibility='collapsed',
        )
        if gemini_key:
            st.session_state['gemini_api_key'] = gemini_key
            st.success('✅ API key aktif', icon='🔑')

    return {
        'ticker':        ticker_input,
        'start_date':    start_date.strftime('%Y-%m-%d'),
        'end_date':      end_date.strftime('%Y-%m-%d'),
        'timeframe':     timeframe,
        'total_capital': float(total_capital),
        'analyze':       analyze,
        'emiten_df':     emiten_df,
    }


def _load_emiten_list_safe() -> Optional[pd.DataFrame]:
    """Muat daftar emiten dengan penanganan error graceful."""
    try:
        from data.emiten_sync import load_emiten_list
        return load_emiten_list()
    except Exception as e:
        logger.warning(f'Gagal memuat daftar emiten: {e}')
        return None


def _render_ipo_banner_sidebar(emiten_df: Optional[pd.DataFrame]) -> None:
    """Tampilkan banner saham IPO baru di sidebar."""
    if emiten_df is None or emiten_df.empty:
        return
    try:
        from analysis.ipo_detector import get_new_listings
        new_listings = get_new_listings(emiten_df, days=7)
        if new_listings:
            st.warning(f'🆕 {len(new_listings)} saham baru listing minggu ini!')
            for emiten in new_listings[:3]:
                if st.button(
                    f'{emiten.ticker} — {emiten.name[:20]}',
                    key=f'ipo_btn_{emiten.ticker}',
                    use_container_width=True,
                ):
                    st.session_state['ticker_override'] = emiten.ticker
                    st.rerun()
    except Exception:
        pass


# ===========================================================================
# HEADER — Task 19.1, 19.2
# ===========================================================================

def render_header(result: AnalysisResult) -> None:
    """Render header halaman utama dengan info saham dan badge."""
    regime_key   = result.market_regime.regime.value
    regime_color = _REGIME_COLORS.get(regime_key, _COLOR_NETRAL)
    data_age_label = _DATA_AGE_LABELS.get(result.data_age, result.data_age.value)

    st.markdown(f'### {result.company_name}')
    st.caption(result.ticker)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Harga', format_rupiah(result.current_price))
    col2.metric(
        'Perubahan',
        f'{result.price_change_pct:+.2f}%',
        delta=result.price_change_pct,
        delta_color='normal',
    )
    col3.markdown(
        f'**Kondisi Pasar**  \n'
        f'<span style="background:{regime_color};color:white;padding:3px 8px;'
        f'border-radius:4px;font-size:0.85em;">{regime_key.replace("_"," ")}</span>',
        unsafe_allow_html=True,
    )
    with col4:
        st.markdown(
            f'**Data Tersedia**  \n'
            f'<span style="background:#424242;color:white;padding:3px 8px;'
            f'border-radius:4px;font-size:0.85em;">{data_age_label} ({result.days_available} hari)</span>',
            unsafe_allow_html=True,
        )
        st.caption(f'Dianalisis: {result.analysis_timestamp}')

    # Task 19.2 — IPO Banner kondisional
    if result.data_age in (DataAgeClassification.IPO_NEW, DataAgeClassification.IPO_PARTIAL):
        st.warning(
            f'⚠️ Saham IPO / Data Terbatas — Klasifikasi: {result.data_age.value}. '
            'Beberapa indikator teknikal tidak tersedia karena data historis belum mencukupi.'
        )

    st.divider()


# ===========================================================================
# TAB 1 — Analisis Teknikal (Task 20)
# ===========================================================================

def render_tab_analisis(result: AnalysisResult, df_daily: pd.DataFrame) -> None:
    """Render Tab 1: Grafik teknikal interaktif."""
    from visualization.charts import build_combined_chart
    from analysis.indicators import get_nan_indicator_messages

    # Task 20.1 — Grafik Plotly
    fig = build_combined_chart(
        df=df_daily,
        sr_result=result.sr_result,
        signal_daily=result.signal_daily,
        divergences=result.divergences,
        data_age=result.data_age,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Task 20.2 — Pola candlestick terdeteksi
    pattern_names = {
        'CDL_HAMMER':       'Hammer (Bullish Reversal)',
        'CDL_SHOOTING_STAR': 'Shooting Star (Bearish Reversal)',
        'CDL_DOJI_10_0.1':  'Doji (Ketidakpastian)',
        'CDL_ENGULFING':    'Engulfing',
    }
    detected: list[str] = []
    if not df_daily.empty:
        last = df_daily.iloc[-1]
        for col, name in pattern_names.items():
            if col in df_daily.columns:
                val = last.get(col, 0)
                if val is not None and not pd.isna(val) and val != 0:
                    detected.append(name)
    if detected:
        st.caption(f'Pola terdeteksi: {", ".join(detected)}')

    # Task 20.3 — Tabel ringkasan indikator
    with st.expander('Nilai Indikator', expanded=False):
        _render_indicator_summary(df_daily)

    # Task 8.9 — Indikator tidak tersedia
    nan_messages = get_nan_indicator_messages(df_daily, result.data_age)
    if nan_messages:
        with st.expander(f'{len(nan_messages)} indikator tidak tersedia', expanded=False):
            for msg in nan_messages:
                st.info(msg)


def _render_indicator_summary(df: pd.DataFrame) -> None:
    """Tampilkan tabel ringkasan nilai indikator terakhir."""
    if df.empty:
        return
    last = df.iloc[-1]

    def _get_val_by_prefix(prefix: str) -> str:
        """Cari kolom berdasarkan prefix dan kembalikan nilai terakhir."""
        for col in df.columns:
            if col.startswith(prefix + '_') and col in last.index:
                val = last.get(col)
                if val is not None and not pd.isna(val):
                    return f'{float(val):.2f}'
        return 'N/A'

    def _get_val_exact(col: str) -> str:
        """Ambil nilai kolom eksak."""
        if col not in df.columns:
            return 'N/A'
        val = last.get(col)
        return f'{float(val):.2f}' if val is not None and not pd.isna(val) else 'N/A'

    rows = [
        {'Indikator': 'RSI(14)',     'Nilai': _get_val_exact('RSI_14')},
        {'Indikator': 'MACD',        'Nilai': _get_val_exact('MACD_12_26_9')},
        {'Indikator': 'MACD Signal', 'Nilai': _get_val_exact('MACDs_12_26_9')},
        {'Indikator': 'ADX(14)',     'Nilai': _get_val_exact('ADX_14')},
        {'Indikator': '+DI',         'Nilai': _get_val_exact('DMP_14')},
        {'Indikator': '-DI',         'Nilai': _get_val_exact('DMN_14')},
        {'Indikator': 'MA20',        'Nilai': _get_val_exact('SMA_20')},
        {'Indikator': 'MA50',        'Nilai': _get_val_exact('SMA_50')},
        {'Indikator': 'MA200',       'Nilai': _get_val_exact('SMA_200')},
        {'Indikator': 'BB Upper',    'Nilai': _get_val_by_prefix('BBU')},
        {'Indikator': 'BB Middle',   'Nilai': _get_val_by_prefix('BBM')},
        {'Indikator': 'BB Lower',    'Nilai': _get_val_by_prefix('BBL')},
        {'Indikator': 'ATR(14)',     'Nilai': _get_val_exact('ATRr_14')},
        {'Indikator': 'MFI(14)',     'Nilai': _get_val_exact('MFI_14')},
        {'Indikator': 'Stoch %K',    'Nilai': _get_val_by_prefix('STOCHk')},
        {'Indikator': 'Stoch %D',    'Nilai': _get_val_by_prefix('STOCHd')},
        {'Indikator': 'Williams %R', 'Nilai': _get_val_exact('WILLR_14')},
        {'Indikator': 'CCI(20)',     'Nilai': _get_val_exact('CCI_20')},
    ]
    # Filter baris yang N/A agar tabel tidak terlalu panjang
    rows = [r for r in rows if r['Nilai'] != 'N/A']
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption('Belum ada nilai indikator yang tersedia.')


# ===========================================================================
# TAB 2 — Rekomendasi (Task 21)
# ===========================================================================

def render_tab_rekomendasi(result: AnalysisResult) -> None:
    """Render Tab 2: Kartu rekomendasi 3 horizon."""
    if all_signals_neutral(result.recommendations):
        st.caption('Belum ada sinyal yang cukup kuat. Pantau perkembangan pasar.')

    cols = st.columns(3)
    for i, card in enumerate(result.recommendations):
        with cols[i]:
            with st.container(border=True):
                signal_color = {
                    SignalDirection.BELI:   _COLOR_BELI,
                    SignalDirection.JUAL:   _COLOR_JUAL,
                    SignalDirection.NETRAL: _COLOR_NETRAL,
                }.get(card.signal, _COLOR_NETRAL)

                st.markdown(f'### {card.horizon}  \n*{card.horizon_detail}*')
                signal_label = card.signal.value
                strength_label = card.signal_strength.value
                st.markdown(f'**{signal_label}** · {strength_label}')
                st.markdown(
                    f'<div style="height:3px;background:{signal_color};border-radius:2px;margin-bottom:12px;"></div>',
                    unsafe_allow_html=True,
                )
                st.metric('Confidence', f'{card.confidence_pct:.0f}%')

                if card.signal != SignalDirection.NETRAL and card.entry_price > 0:
                    st.markdown(f'**Entry:** {format_rupiah(card.entry_price)}')
                    st.markdown(f'**Stop Loss:** {format_rupiah(card.stop_loss)}')
                    st.markdown(f'**TP1:** {format_rupiah(card.tp1)}')
                    st.markdown(f'**TP2:** {format_rupiah(card.tp2)}')
                    st.markdown(f'**TP3:** {format_rupiah(card.tp3)}')
                    rr_text = f'1:{card.rr_ratio:.1f}' if card.rr_ratio is not None else 'N/A'
                    st.markdown(f'**R/R Ratio:** {rr_text}')

                st.markdown(f'**Durasi:** {card.holding_duration}')
                if card.timeframe_alignment:
                    st.caption('↑ Multi-timeframe selaras')

                if card.explanation and card.explanation != ['Belum ada sinyal yang cukup kuat saat ini.']:
                    with st.expander('Detail Sinyal', expanded=False):
                        for line in card.explanation:
                            if line.startswith('DIM:'):
                                # Header dimensi — tampilkan sebagai label uppercase
                                clean = line[4:].strip()
                                st.markdown(
                                    f'<div style="font-size:0.72rem;font-weight:700;'
                                    f'color:var(--text-color,#111);'
                                    f'text-transform:uppercase;letter-spacing:0.04em;'
                                    f'margin-top:10px;margin-bottom:2px;'
                                    f'border-bottom:1px solid rgba(128,128,128,0.15);'
                                    f'padding-bottom:2px;">{clean}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                # Baris indikator
                                clean = line.strip()
                                st.markdown(
                                    f'<div style="font-size:0.80rem;'
                                    f'color:var(--text-color,#333);'
                                    f'padding:2px 0 2px 10px;line-height:1.6;">'
                                    f'— {clean}</div>',
                                    unsafe_allow_html=True,
                                )
                elif card.signal == SignalDirection.NETRAL:
                    st.caption('Belum ada sinyal yang cukup kuat saat ini.')

    st.divider()
    st.markdown('**Interpretasi**')
    st.markdown(result.combined_narrative)


# ===========================================================================
# TAB 3 — Manajemen Risiko (Task 22)
# ===========================================================================

def render_tab_risiko(result: AnalysisResult) -> None:
    """Render Tab 3: Tabel SL/TP, R/R ratio, dan position sizing."""
    risk   = result.risk_result
    signal = result.signal_daily

    if signal.direction == SignalDirection.NETRAL or risk.entry_price == 0:
        st.info('Tidak ada sinyal aktif. Manajemen risiko tidak tersedia saat ini.')
        return

    st.markdown('**Level Harga**')

    # Task 24.5 — Catatan ATR
    if risk.sl_method in ('S/R', 'Fallback 5%'):
        st.caption(f'ℹ️ SL dihitung dari {risk.sl_method} (ATR tidak tersedia)')

    entry = risk.entry_price

    def _jarak(price: float) -> str:
        if entry > 0:
            return f'{abs(price - entry) / entry * 100:.2f}%'
        return 'N/A'

    def _rr(ratio) -> str:
        return f'1:{ratio:.1f}' if ratio is not None else 'N/A'

    # Task 22.1 — Tabel SL/TP
    rows = [
        {'Level': 'Entry',     'Harga': format_rupiah(entry),      'Jarak %': '0.00%',           'R/R Ratio': '—'},
        {'Level': 'Stop Loss', 'Harga': format_rupiah(risk.stop_loss), 'Jarak %': _jarak(risk.stop_loss), 'R/R Ratio': '1:1'},
        {'Level': 'TP1',       'Harga': format_rupiah(risk.tp1),    'Jarak %': _jarak(risk.tp1),  'R/R Ratio': _rr(risk.rr_ratio_tp1)},
        {'Level': 'TP2',       'Harga': format_rupiah(risk.tp2),    'Jarak %': _jarak(risk.tp2),  'R/R Ratio': _rr(risk.rr_ratio_tp2)},
        {'Level': 'TP3',       'Harga': format_rupiah(risk.tp3),    'Jarak %': _jarak(risk.tp3),  'R/R Ratio': _rr(risk.rr_ratio_tp3)},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Task 22.2 — Visualisasi R/R ratio
    st.markdown('**Risk / Reward**')
    max_rr = max(
        r for r in [risk.rr_ratio_tp1, risk.rr_ratio_tp2, risk.rr_ratio_tp3]
        if r is not None
    ) if any(r is not None for r in [risk.rr_ratio_tp1, risk.rr_ratio_tp2, risk.rr_ratio_tp3]) else 1.0

    for label, rr in [('TP1', risk.rr_ratio_tp1), ('TP2', risk.rr_ratio_tp2), ('TP3', risk.rr_ratio_tp3)]:
        if rr is not None:
            progress_val = min(rr / (max_rr * 1.2), 1.0)
            st.markdown(f'**{label}** — R/R 1:{rr:.1f}')
            st.progress(progress_val)

    # Task 22.3 & 22.4 — Position sizing (hanya jika modal diisi)
    if risk.position_size_lots is not None and risk.position_size_lots > 0:
        st.markdown('**Position Sizing**')
        col1, col2, col3 = st.columns(3)
        col1.metric('Jumlah Lot', f'{risk.position_size_lots} lot')
        col2.metric('Modal Digunakan', format_rupiah(risk.capital_at_risk or 0))
        # Hitung risiko per transaksi dengan aman
        try:
            risiko = abs(entry - risk.stop_loss) * risk.position_size_lots * 100
            risiko_str = format_rupiah(risiko)
        except (TypeError, ValueError):
            risiko_str = 'N/A'
        col3.metric('Risiko per Transaksi', risiko_str)
        st.caption('1 lot = 100 lembar saham · Risiko maksimum 2% dari total modal')


# ===========================================================================
# TAB 4 — IPO Radar (Task 23)
# ===========================================================================

def render_tab_ipo_radar(emiten_df: Optional[pd.DataFrame]) -> None:
    """Render Tab 4: Tabel saham IPO dalam 30 hari terakhir."""
    st.markdown('**Listing Baru — 30 Hari Terakhir**')

    if emiten_df is None or emiten_df.empty:
        st.warning('Daftar emiten tidak tersedia. Periksa koneksi internet.')
        return

    try:
        from analysis.ipo_detector import get_ipo_radar
        ipo_df = get_ipo_radar(emiten_df, days=30)
    except Exception as e:
        st.error(f'Gagal memuat data IPO Radar: {e}')
        return

    if ipo_df is None or ipo_df.empty:
        st.info('Tidak ada saham baru listing dalam 30 hari terakhir.')
        return

    # Task 23.1 — Tabel IPO Radar
    display_cols = [c for c in ['ticker', 'name', 'listing_date', 'sector', 'days_listed'] if c in ipo_df.columns]
    col_labels = {
        'ticker': 'Ticker', 'name': 'Nama Perusahaan',
        'listing_date': 'Tanggal Listing', 'sector': 'Sektor', 'days_listed': 'Hari Listed',
    }

    display_df = ipo_df[display_cols].rename(columns=col_labels)

    # Task 23.2 — on_select callback
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select='rerun',
        selection_mode='single-row',
    )

    if event and event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        if selected_idx < len(ipo_df):
            selected_ticker = ipo_df.iloc[selected_idx]['ticker']
            st.session_state['ticker_override'] = selected_ticker
            st.info(f'Ticker {selected_ticker} dipilih. Klik tombol Analisis di sidebar untuk menganalisis.')


# ===========================================================================
# GEMINI AI ASSISTANT — Section utama (di bawah disclaimer)
# ===========================================================================

def render_gemini_chat(
    analysis_result=None,
    df_daily=None,
) -> None:
    """
    Render section Gemini AI Assistant di halaman utama (di bawah disclaimer).

    Menampilkan riwayat percakapan lengkap, tombol pertanyaan cepat,
    dan chat input untuk berinteraksi dengan StockMind AI.
    """
    from ai.gemini_assistant import (
        init_gemini,
        build_analysis_context,
        chat_with_gemini,
        extract_urls,
    )

    st.divider()
    st.markdown('**Tanya Analis**')

    api_key = st.session_state.get('gemini_api_key', '').strip()

    if not api_key:
        st.info(
            '💡 **Aktifkan AI Assistant** dengan memasukkan Gemini API key di sidebar.\n\n'
            'API key gratis tersedia di [Google AI Studio](https://aistudio.google.com/apikey) — '
            'tidak perlu kartu kredit.',
        )
        return

    model_cache_key = f'gemini_model_{api_key[:8]}'
    if model_cache_key not in st.session_state:
        with st.spinner('Menghubungkan ke Gemini AI...'):
            model = init_gemini(api_key)
        if model is None:
            st.error(
                '❌ Gagal menginisialisasi Gemini. Pastikan:\n'
                '1. `google-generativeai` sudah terinstall: `pip install google-generativeai`\n'
                '2. API key valid dan aktif'
            )
            return
        st.session_state[model_cache_key] = model

    model = st.session_state[model_cache_key]

    # Gunakan session_state terpusat — sama dengan sidebar
    if 'gemini_chat_history' not in st.session_state:
        st.session_state['gemini_chat_history'] = []

    history: list[dict] = st.session_state['gemini_chat_history']

    # Konteks analisis
    analysis_context = ''
    if analysis_result is not None:
        ctx_key = (
            f'gemini_ctx_'
            f'{getattr(analysis_result, "ticker", "")}_'
            f'{getattr(analysis_result, "analysis_timestamp", "")}'
        )
        if ctx_key not in st.session_state:
            st.session_state[ctx_key] = build_analysis_context(analysis_result, df_daily)
        analysis_context = st.session_state[ctx_key]

    # --- Riwayat percakapan lengkap ---
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        if history:
            st.caption(f'{len(history) // 2} pesan')
    with col_reset:
        if st.button('Hapus riwayat', key='main_chat_reset'):
            st.session_state['gemini_chat_history'] = []
            st.rerun()

    # Tampilkan riwayat
    if not history:
        with st.chat_message('assistant', avatar='🤖'):
            if analysis_result is not None:
                ticker = getattr(analysis_result, 'ticker', '')
                st.markdown(f'Konteks analisis **{ticker}** sudah dimuat. Tanyakan apa saja.')
            else:
                st.markdown('Tanyakan tentang indikator teknikal atau paste URL berita.')
    else:
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('parts', [''])[0] if msg.get('parts') else ''
            avatar = '🤖' if role == 'model' else '👤'
            display_role = 'assistant' if role == 'model' else 'user'
            with st.chat_message(display_role, avatar=avatar):
                st.markdown(content)

    # --- Tombol pertanyaan cepat ---
    if not history and analysis_result is not None:
        st.caption('Pertanyaan cepat:')
        quick_cols = st.columns(3)
        quick_questions = [
            'Jelaskan kondisi RSI dan MACD saat ini',
            'Interpretasi sinyal Bollinger Bands',
            'Cara membaca ADX dan DI',
            'Rekomendasi jangka pendek',
            'Ada divergensi saat ini?',
            'Tips manajemen risiko',
        ]
        for i, q in enumerate(quick_questions):
            with quick_cols[i % 3]:
                if st.button(q, key=f'main_quick_q_{i}', use_container_width=True):
                    st.session_state['gemini_main_pending'] = q
                    st.rerun()

    # Handle pertanyaan cepat
    pending_main = st.session_state.pop('gemini_main_pending', None)

    # --- Input chat utama ---
    if analysis_result is not None:
        ticker_clean = getattr(analysis_result, 'ticker', '').replace('.JK', '')
        placeholder = f'Tanya tentang {ticker_clean}, atau paste URL berita untuk dianalisis...'
    else:
        placeholder = 'Tanyakan tentang indikator teknikal, atau paste URL berita saham...'

    user_input = st.chat_input(placeholder, key='main_chat_input')
    final_input = user_input or pending_main

    if final_input:
        with st.chat_message('user', avatar='👤'):
            st.markdown(final_input)

        urls_found = extract_urls(final_input)
        if urls_found:
            st.caption(
                f'🔗 Membaca {len(urls_found)} URL: '
                f'{", ".join(u[:60] + "..." if len(u) > 60 else u for u in urls_found)}'
            )

        with st.chat_message('assistant', avatar='🤖'):
            with st.spinner('Menganalisis...'):
                response_text, updated_history = chat_with_gemini(
                    model=model,
                    user_message=final_input,
                    chat_history=history,
                    analysis_context=analysis_context,
                )
            st.markdown(response_text)

        st.session_state['gemini_chat_history'] = updated_history


# ===========================================================================
# FOOTER — Task 23.3
# ===========================================================================

def render_footer() -> None:
    """Render footer dengan disclaimer investasi."""
    st.divider()
    st.caption(
        'Alat bantu analisis teknikal. Bukan saran investasi. '
        'Selalu lakukan riset mandiri sebelum mengambil keputusan.'
    )


# ===========================================================================
# MAIN — Orkestrasi utama aplikasi
# ===========================================================================

def main() -> None:
    """Fungsi utama aplikasi StockMomentum ID."""

    # CSS global — transparansi dan tipografi profesional
    st.markdown(
        """
<style>
/* ================================================================
   SIDEBAR — warna, kontras, dan keterbacaan
   ================================================================ */

/* Background sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(14, 17, 23, 0.92);
    border-right: 1px solid rgba(255,255,255,0.10);
}

/* Teks umum sidebar (paragraf, span, div) — TIDAK pakai wildcard * */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(input):not(textarea),
[data-testid="stSidebar"] div:not(input):not(textarea),
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E8EAF0;
}

/* Judul / bold text sidebar */
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Label widget (Ticker, Dari, Sampai, Modal, dll.) */
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #B0B8CC !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ----------------------------------------------------------------
   INPUT FIELDS — background gelap agar teks putih terbaca
   Ini kunci utama: jangan biarkan background default (putih)
   ---------------------------------------------------------------- */

/* Text input (Ticker, API Key) */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #1E2433 !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #F0F2F8 !important;
    border-radius: 5px !important;
    caret-color: #7EB8F7 !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #4A90D9 !important;
    box-shadow: 0 0 0 2px rgba(74,144,217,0.20) !important;
    outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: rgba(176,184,204,0.45) !important;
}

/* Number input (Modal) */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background-color: #1E2433 !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #F0F2F8 !important;
    border-radius: 5px !important;
    caret-color: #7EB8F7 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {
    border-color: #4A90D9 !important;
    box-shadow: 0 0 0 2px rgba(74,144,217,0.20) !important;
}

/* Date input (Dari, Sampai) */
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background-color: #1E2433 !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #F0F2F8 !important;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] input:focus {
    border-color: #4A90D9 !important;
}

/* Selectbox (Timeframe) */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: #1E2433 !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
    color: #F0F2F8 !important;
}

/* Ikon panah selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: #B0B8CC !important;
}

/* ----------------------------------------------------------------
   ELEMEN UI LAIN DI SIDEBAR
   ---------------------------------------------------------------- */

/* Caption / teks kecil */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: rgba(176,184,204,0.60) !important;
    font-size: 0.75rem !important;
}

/* Link */
[data-testid="stSidebar"] a {
    color: #7EB8F7 !important;
    text-decoration: none !important;
}
[data-testid="stSidebar"] a:hover {
    color: #A8D0FF !important;
    text-decoration: underline !important;
}

/* Divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 0.6rem 0 !important;
}

/* Tombol Analisis (primary) */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background-color: #2563EB !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background-color: #1D4ED8 !important;
}

/* Tombol secondary (IPO banner) */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #E8EAF0 !important;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background-color: rgba(255,255,255,0.14) !important;
}

/* Warning box (IPO banner) */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: rgba(255,193,7,0.12) !important;
    border: 1px solid rgba(255,193,7,0.30) !important;
    border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p {
    color: #FFE082 !important;
}

/* Success box (API key aktif) */
[data-testid="stSidebar"] [data-testid="stNotification"] p {
    color: #A5D6A7 !important;
}

/* ================================================================
   MAIN CONTENT — container, metric, tab, expander
   ================================================================ */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(255,255,255,0.10) !important;
    border-radius: 6px !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}

[data-testid="stTabs"] button {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stCaptionContainer"] {
    color: rgba(255,255,255,0.40) !important;
}

hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 0.75rem 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
    # Task 18.3 — Cache invalidation saat ticker berubah
    params = render_sidebar()
    ticker = params['ticker']

    if 'last_ticker' not in st.session_state or st.session_state['last_ticker'] != ticker:
        st.session_state.pop('analysis_result', None)
        st.session_state.pop('df_daily_cache', None)
        st.session_state['last_ticker'] = ticker

    # Halaman awal sebelum analisis
    if not params['analyze'] and 'analysis_result' not in st.session_state:
        st.markdown('## StockMomentum ID')
        st.markdown('Analisis teknikal saham BEI — masukkan ticker di sidebar untuk memulai.')
        st.divider()
        render_tab_ipo_radar(params.get('emiten_df'))
        render_footer()
        render_gemini_chat()
        return

    # Task 18.4 — Validasi input ticker
    if params['analyze']:
        if not ticker:
            st.error('⚠️ Masukkan kode saham terlebih dahulu. Contoh: BBCA, TLKM, atau ^JKSE')
            render_footer()
            return

        # Task 24.1 — Wrap analisis dalam try/except utama
        with st.spinner(f'Menganalisis {ticker}...'):
            try:
                from recommendation.engine import run_full_analysis

                # Normalisasi ticker: tambahkan .JK jika bukan indeks
                ticker_yf = ticker
                if not ticker.startswith('^') and not ticker.endswith('.JK'):
                    ticker_yf = ticker + '.JK'

                result = run_full_analysis(
                    ticker=ticker_yf,
                    start_date=params['start_date'],
                    end_date=params['end_date'],
                    total_capital=params['total_capital'],
                )

                # Simpan ke session_state
                st.session_state['analysis_result'] = result

                # Ambil df_daily untuk grafik — gunakan ulang data dari engine
                # dengan menghitung ulang indikator dari data harian yang sudah di-cache
                from data.fetcher import fetch_ohlcv
                from analysis.indicators import calculate_all_indicators
                try:
                    df_raw = fetch_ohlcv(ticker_yf, params['start_date'], params['end_date'], '1d')
                    df_daily = calculate_all_indicators(df_raw, result.data_age)
                except Exception:
                    df_daily = pd.DataFrame()
                st.session_state['df_daily_cache'] = df_daily

            except ValueError as e:
                err_msg = str(e)
                # Task 24.2 — Ticker tidak ditemukan
                if 'tidak ditemukan' in err_msg.lower() or 'tidak tersedia' in err_msg.lower():
                    st.error(
                        f'❌ Kode saham **{ticker}** tidak ditemukan. '
                        'Pastikan format KODE.JK (contoh: BBCA.JK) atau gunakan ^JKSE untuk IHSG.'
                    )
                # Task 24.3 — Data kosong
                elif 'tidak ada data' in err_msg.lower() or 'data ohlcv' in err_msg.lower():
                    st.error(
                        f'❌ Tidak ada data untuk **{ticker}** pada periode yang dipilih. '
                        'Coba perluas rentang tanggal atau periksa kode saham.'
                    )
                else:
                    st.error(f'❌ Terjadi kesalahan saat menganalisis {ticker}: {err_msg}')
                render_footer()
                return

            except Exception as e:
                st.error(
                    f'❌ Terjadi kesalahan tidak terduga saat menganalisis **{ticker}**.  \n'
                    f'Detail: {str(e)}  \n\n'
                    'Coba periksa koneksi internet atau coba lagi beberapa saat.'
                )
                logger.exception(f'Error saat menganalisis {ticker}')
                render_footer()
                return

    # Ambil hasil dari session_state
    result: AnalysisResult = st.session_state.get('analysis_result')
    df_daily: pd.DataFrame = st.session_state.get('df_daily_cache', pd.DataFrame())

    if result is None:
        render_footer()
        return

    # Task 19.1 — Render header
    render_header(result)

    # Task 12.3 — 4 Tab utama
    tab1, tab2, tab3, tab4 = st.tabs([
        'Teknikal',
        'Rekomendasi',
        'Risiko',
        'IPO Radar',
    ])

    with tab1:
        render_tab_analisis(result, df_daily)

    with tab2:
        render_tab_rekomendasi(result)

    with tab3:
        render_tab_risiko(result)

    with tab4:
        render_tab_ipo_radar(params.get('emiten_df'))

    render_footer()
    render_gemini_chat(
        analysis_result=result,
        df_daily=df_daily,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
else:
    # Streamlit memanggil file langsung, bukan via __main__
    main()
