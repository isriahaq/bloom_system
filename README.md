# Sistem Klasifikasi Taksonomi Bloom

Sistem berbasis LLM (Large Language Model) untuk mengklasifikasikan butir soal ujian ke dalam level Taksonomi Bloom secara otomatis, dilengkapi anotasi warna pada PDF hasil dan ekspor CSV.

**Dikembangkan sebagai bagian dari skripsi:**
> *PENGEMBANGAN SISTEM KLASIFIKASI TAKSONOMI BLOOM BERBASIS LARGE LANGUAGE MODEL DENGAN ARSITEKTUR CASCADING FALLBACK DAN ANOTASI VISUAL PDF*

---

## Fitur Utama

- Ekstraksi butir soal otomatis dari file PDF berbasis teks
- Klasifikasi tingkat kognitif (C1–C6) dan kategori (LOTS / MOTS / HOTS) menggunakan Gemini API
- Arsitektur **Cascading Fallback** antar-varian model Gemini (Pure Cloud)
- Deteksi anomali opsi jawaban pilihan ganda
- Anotasi warna pada PDF hasil (Hijau = LOTS, Kuning = MOTS, Oranye = HOTS)
- Dasbor analitik distribusi kognitif
- Ekspor hasil ke PDF teranotasi dan CSV

---

## Teknologi

| Komponen | Teknologi |
|---|---|
| Bahasa | Python 3.13 |
| LLM | Google Gemini API (gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash) |
| Web Interface | Streamlit ≥ 1.35 |
| Pemrosesan PDF | PyMuPDF (fitz) |
| Validasi Output | Pydantic 2.x |

---

## Arsitektur Cascading Fallback

```
evaluate_batch(questions)
   └─► _call_gemini_cascade()
           ├─ gemini-3.7-flash  (Prioritas 1)
           ├─ gemini-3.6-flash  (Prioritas 2 — fallback 429/5xx)
           └─ gemini-3.5-flash  (Prioritas 3 — fallback terakhir)
```

---

## Struktur Folder

```
bloom_system/
├── app.py                  # Antarmuka web utama (Streamlit)
├── config.py               # Konfigurasi terpusat
├── requirements.txt        # Daftar dependensi Python
├── .gitignore              # File yang dikecualikan dari Git
├── .streamlit/
│   └── secrets.toml        # Brankas kunci API (tidak diunggah ke GitHub)
├── modules/
│   ├── llm_router.py       # Router AI — Cascading Fallback Gemini
│   ├── pdf_processor.py    # Ekstraksi & segmentasi teks PDF
│   └── visual_annotator.py # Anotasi warna pada halaman PDF
└── utils/
    └── logger.py           # Logging terpusat (loguru)
```

---

## Cara Menjalankan Lokal

1. **Clone repositori:**
   ```bash
   git clone https://github.com/USERNAME/NAMA-REPO.git
   cd NAMA-REPO
   ```

2. **Pasang dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Buat file kunci API:**
   Buat file `.streamlit/secrets.toml` dengan isi:
   ```toml
   GEMINI_API_KEY = "kunci-api-anda"
   ```

4. **Jalankan:**
   ```bash
   streamlit run app.py
   ```

---

## Cara Memakai Aplikasi

1. Unggah file PDF soal ujian berbasis teks (bukan scan/gambar)
2. Klik **Mulai Analisis Dokumen**
3. Tunggu proses selesai (tergantung jumlah soal dan koneksi API)
4. Lihat dasbor distribusi kognitif
5. Unduh PDF teranotasi dan/atau ringkasan CSV

---

## Privasi & Keamanan

- Sistem **tidak menyimpan** berkas PDF yang diunggah setelah sesi selesai
- Kunci API disimpan di brankas Streamlit, tidak di dalam kode
- File PDF soal ujian **tidak** disimpan di repositori ini

---

## Sitasi

Jika menggunakan sistem ini untuk penelitian, harap sitasi:

> [Nama Penulis]. (2026). *Pengembangan Sistem Klasifikasi Taksonomi Bloom Berbasis Large Language Model dengan Arsitektur Cascading Fallback dan Anotasi Visual PDF*. [Institusi]. 

---

## Lisensi

Untuk keperluan akademis. Hubungi penulis untuk penggunaan lainnya.
