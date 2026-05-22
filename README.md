# StockMomentum ID

Aplikasi analisis teknikal saham Bursa Efek Indonesia (BEI) berbasis Streamlit dengan sistem sinyal 5 dimensi, rekomendasi multi-horizon, manajemen risiko otomatis, dan AI Assistant berbasis Gemini.

## Fitur Utama

- **Analisis 5 Dimensi** — Tren, Momentum, Volatilitas, Volume, dan Pola Candlestick
- **Rekomendasi Multi-Horizon** — Jangka pendek (1–3 bulan), menengah (3–12 bulan), dan panjang (1–3 tahun)
- **Deteksi Kondisi Pasar** — BULL_TREND, BEAR_TREND, SIDEWAYS, BREAKOUT
- **Manajemen Risiko Otomatis** — Stop Loss, Take Profit, dan Position Sizing berbasis aturan 2%
- **Deteksi Divergensi** — Bullish/Bearish Regular dan Hidden Divergence
- **Level Support & Resistance** — Statis, Dinamis (MA), Fibonacci, dan Psikologis
- **IPO Radar** — Pantau saham baru listing dalam 30 hari terakhir
- **Penanganan Saham IPO** — Analisis adaptif berdasarkan ketersediaan data historis
- **AI Assistant (StockMind)** — Chat berbasis Gemini untuk interpretasi indikator dan analisis berita

## Instalasi

### Prasyarat

- Python 3.9 atau lebih baru
- pip (Python package manager)

### Langkah Instalasi

1. Clone atau unduh repositori ini:
   ```bash
   git clone <url-repositori>
   cd stock-momentum-id
   ```

2. (Opsional) Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. Install semua dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

## Cara Penggunaan

1. Masukkan kode saham BEI pada sidebar (contoh: `BBCA`, `TLKM`, atau `^JKSE` untuk IHSG)
2. Pilih rentang tanggal analisis (default: 2 tahun terakhir)
3. Pilih timeframe tampilan grafik (Harian/Mingguan/Bulanan)
4. Masukkan total modal (opsional) untuk kalkulasi position sizing
5. Klik tombol **Analisis**
6. Navigasi antar tab untuk melihat hasil analisis:
   - **Teknikal** — Grafik interaktif dengan semua indikator
   - **Rekomendasi** — Tiga kartu rekomendasi per horizon investasi
   - **Risiko** — Tabel SL/TP, R/R Ratio, dan position sizing
   - **IPO Radar** — Daftar saham baru listing 30 hari terakhir

### AI Assistant (StockMind)

Fitur chat AI tersedia di bagian bawah halaman utama. Untuk mengaktifkan:

1. Dapatkan API key gratis di [Google AI Studio](https://aistudio.google.com/apikey)
2. Masukkan API key pada field **Gemini API Key** di sidebar
3. Gunakan chat box untuk:
   - Menanyakan penjelasan indikator teknikal
   - Menginterpretasikan sinyal dan rekomendasi
   - Paste URL berita untuk dianalisis bersama data teknikal

## Struktur Proyek

```
stock-momentum-id/
├── app.py                          # Entry point aplikasi Streamlit
├── models.py                       # Dataclass dan enum model data
├── requirements.txt                # Daftar dependensi Python
├── README.md                       # Dokumentasi ini
├── .gitignore
├── ai/
│   ├── __init__.py
│   └── gemini_assistant.py         # AI Assistant berbasis Gemini
├── data/
│   ├── __init__.py
│   ├── fetcher.py                  # Pengambilan data OHLCV dari yfinance
│   ├── emiten_sync.py              # Sinkronisasi daftar emiten BEI
│   └── cache/                      # Auto-generated, tidak di-commit
│       ├── emiten_list.csv
│       └── last_sync.txt
├── analysis/
│   ├── __init__.py
│   ├── market_regime.py            # Deteksi kondisi pasar
│   ├── indicators.py               # Perhitungan semua indikator teknikal
│   ├── signals.py                  # Mesin sinyal 5 dimensi
│   ├── divergence.py               # Deteksi divergensi harga-indikator
│   ├── support_resistance.py       # Kalkulasi level Support & Resistance
│   └── ipo_detector.py             # Penanganan saham IPO dan baru listing
├── recommendation/
│   ├── __init__.py
│   ├── scorer.py                   # Agregasi skor dan filter kondisi pasar
│   ├── risk_calculator.py          # Kalkulasi SL, TP, dan position sizing
│   └── engine.py                   # Mesin rekomendasi dan narasi
├── visualization/
│   ├── __init__.py
│   └── charts.py                   # Pembuat grafik Plotly interaktif
└── tests/
    ├── __init__.py
    └── test_data_engine.py         # Unit test data engine
```

## Dependensi

| Library | Versi | Kegunaan |
|---|---|---|
| streamlit | >=1.32.0 | Framework UI web |
| yfinance | >=0.2.37 | Data harga saham |
| pandas | >=2.1.0 | Manipulasi data |
| numpy | >=1.26.0 | Komputasi numerik |
| pandas-ta | >=0.3.14b | Indikator teknikal |
| plotly | >=5.19.0 | Grafik interaktif |
| requests | >=2.31.0 | HTTP client |
| beautifulsoup4 | >=4.12.0 | Parsing HTML IDX & berita |
| lxml | >=5.1.0 | HTML parser |
| scipy | >=1.12.0 | Komputasi ilmiah (S/R) |
| scikit-learn | >=1.4.0 | Machine learning |
| google-generativeai | >=0.8.0 | Gemini AI Assistant |
| pytest | >=7.0.0 | Test runner |
| hypothesis | >=6.0.0 | Property-based testing |

## Disclaimer

Aplikasi ini hanya sebagai alat bantu analisis teknikal. **Bukan merupakan saran investasi.** Selalu lakukan riset mandiri dan konsultasikan dengan penasihat keuangan sebelum mengambil keputusan investasi. Investasi di pasar saham mengandung risiko kehilangan sebagian atau seluruh modal.
