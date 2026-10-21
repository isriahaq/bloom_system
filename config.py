# =============================================================================
# config.py
# Konfigurasi terpusat untuk Sistem Klasifikasi Taksonomi Bloom
# Arsitektur: Cascading Fallback antar-varian model Google Gemini (Pure Cloud)
# =============================================================================
# Semua nilai sensitif (API Key) sebaiknya disimpan di file .env,
# lalu dimuat oleh python-dotenv. Nilai di sini adalah nilai default/fallback.

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Memuat variabel dari file .env jika ada (untuk pengembangan lokal)
load_dotenv()


def _baca_gemini_api_key() -> str:
    """
    Membaca GEMINI_API_KEY secara aman dengan prioritas:
    1. st.secrets (Streamlit Cloud — brankas resmi)
    2. os.getenv (variabel environment / file .env lokal)
    """
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


# =============================================================================
# BAGIAN 1: PATH & DIREKTORI
# =============================================================================

# Direktori root proyek (satu level di atas file ini)
BASE_DIR: Path = Path(__file__).resolve().parent

# Direktori data input (PDF mentah yang akan diproses)
INPUT_DIR: Path = BASE_DIR / "data" / "input"

# Direktori data output (PDF hasil anotasi + file CSV log)
OUTPUT_DIR: Path = BASE_DIR / "data" / "output"

# Pastikan direktori output selalu ada saat modul ini diimpor
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BAGIAN 2: KONFIGURASI GEMINI (LLM Utama — Pure Cloud)
# =============================================================================

# API Key Gemini - dibaca dari st.secrets (Streamlit Cloud) atau .env (lokal)
GEMINI_API_KEY: str = _baca_gemini_api_key()

# Rantai kaskade prioritas model Gemini.
# Sistem akan mencoba model secara berurutan dari atas ke bawah.
# Jika terjadi error (429, 5xx, timeout) maka beban berpindah ke model berikutnya.
GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.7-flash",    # Prioritas 1 — penalaran dan kuota tertinggi
    "gemini-3.6-flash",    # Prioritas 2 — fallback kaskade pertama
    "gemini-3.5-flash",    # Prioritas 3 — fallback kaskade kedua
    "gemini-3.8-flash",    # Alternatif tambahan
    "gemini-2.5-flash",    # Alternatif tambahan
)

# Model Gemini default (digunakan sebagai referensi konfigurasi utama)
GEMINI_MODEL_NAME: str = GEMINI_MODELS[0]

# Batas waktu tunggu respons dari API Gemini (dalam detik)
GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "30"))

# Jumlah maksimal percobaan ulang pada satu model Gemini sebelum kaskade ke model berikutnya
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))


# =============================================================================
# BAGIAN 3: KONFIGURASI KLASIFIKASI BLOOM
# =============================================================================

# Nama semua level Taksonomi Bloom yang dikenali sistem
BLOOM_LEVELS: list[str] = [
    "C1 - Mengingat",       # LOTS (Lower Order Thinking Skills)
    "C2 - Memahami",        # MOTS
    "C3 - Mengaplikasikan", # MOTS
    "C4 - Menganalisis",    # HOTS (Higher Order Thinking Skills)
    "C5 - Mengevaluasi",    # HOTS
    "C6 - Mencipta",        # HOTS
]

# Pengelompokan kategori untuk laporan ringkasan
HOTS_LEVELS: list[str] = ["C4 - Menganalisis", "C5 - Mengevaluasi", "C6 - Mencipta"]
LOTS_LEVELS: list[str] = ["C1 - Mengingat", "C2 - Memahami", "C3 - Mengaplikasikan"]


# =============================================================================
# BAGIAN 4: KONFIGURASI LOGGING & OUTPUT
# =============================================================================

# Nama file CSV tempat hasil klasifikasi dicatat secara otomatis
LOG_CSV_FILENAME: str = "hasil_klasifikasi_bloom.csv"

# Path lengkap file CSV log
LOG_CSV_PATH: Path = OUTPUT_DIR / LOG_CSV_FILENAME

# Nama file log teks (dari loguru)
LOG_FILE_PATH: Path = BASE_DIR / "logs" / "bloom_system.log"

# Level log minimum yang akan ditampilkan/dicatat ("DEBUG", "INFO", "WARNING", "ERROR")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# =============================================================================
# BAGIAN 5: DATACLASS KONFIGURASI (Pendekatan OOP)
# =============================================================================

@dataclass
class AppConfig:
    """
    Kelas konfigurasi terpusat yang mengelompokkan semua parameter sistem.
    Gunakan kelas ini untuk dependency injection ke modul-modul lain,
    sehingga lebih mudah di-mock saat unit testing.
    """

    # --- Path ---
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    input_dir: Path = field(default_factory=lambda: INPUT_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    # --- Gemini (Pure Cloud Cascade) ---
    gemini_api_key: str = field(default_factory=lambda: GEMINI_API_KEY)
    gemini_models: tuple[str, ...] = field(default_factory=lambda: GEMINI_MODELS)
    gemini_model_name: str = field(default_factory=lambda: GEMINI_MODEL_NAME)
    gemini_timeout: int = field(default_factory=lambda: GEMINI_TIMEOUT)
    gemini_max_retries: int = field(default_factory=lambda: GEMINI_MAX_RETRIES)

    # --- Bloom ---
    bloom_levels: list[str] = field(default_factory=lambda: BLOOM_LEVELS)
    hots_levels: list[str] = field(default_factory=lambda: HOTS_LEVELS)
    lots_levels: list[str] = field(default_factory=lambda: LOTS_LEVELS)

    # --- Logging ---
    log_csv_path: Path = field(default_factory=lambda: LOG_CSV_PATH)
    log_file_path: Path = field(default_factory=lambda: LOG_FILE_PATH)
    log_level: str = field(default_factory=lambda: LOG_LEVEL)

    def validate(self) -> None:
        """
        Memvalidasi konfigurasi kritis sebelum sistem dijalankan.
        Akan memunculkan ValueError jika ada konfigurasi yang tidak valid.
        """
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY belum diisi! "
                "Tambahkan ke file .env atau set sebagai environment variable."
            )

    def __repr__(self) -> str:
        """Representasi string yang aman (tidak menampilkan API Key penuh)."""
        masked_key = f"{self.gemini_api_key[:8]}..." if self.gemini_api_key else "(kosong)"
        return (
            f"AppConfig("
            f"gemini_models={self.gemini_models}, "
            f"api_key='{masked_key}', "
            f"output_dir='{self.output_dir}'"
            f")"
        )


# Instance default yang bisa langsung diimpor oleh modul lain
# Contoh penggunaan: from config import app_config
app_config: AppConfig = AppConfig()
