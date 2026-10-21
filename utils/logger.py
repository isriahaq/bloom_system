# =============================================================================
# utils/logger.py
# Utilitas pencatatan (logging) hasil klasifikasi AI ke file CSV
# =============================================================================
# Modul ini bertanggung jawab atas DUA jenis logging:
# 1. Log teks (via loguru) -> untuk debugging dan monitoring real-time
# 2. Log CSV (via pandas/csv) -> rekaman terstruktur untuk analisis hasil skripsi

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

# Impor konfigurasi terpusat
# Gunakan try/except agar modul ini bisa diuji secara independen
try:
    from config import app_config, LOG_CSV_PATH, LOG_FILE_PATH, LOG_LEVEL
except ImportError:
    # Fallback nilai default jika dijalankan langsung tanpa config.py
    LOG_CSV_PATH = Path("data/output/hasil_klasifikasi_bloom.csv")
    LOG_FILE_PATH = Path("logs/bloom_system.log")
    LOG_LEVEL = "INFO"


# =============================================================================
# BAGIAN 1: KOLOM HEADER CSV
# =============================================================================

# Definisi kolom CSV sebagai konstanta agar mudah diubah tanpa refactor besar
CSV_COLUMNS: list[str] = [
    "Timestamp",        # Waktu pemrosesan (format ISO 8601)
    "Nama_File",        # Nama file PDF sumber
    "Nomor_Soal",       # Nomor urut soal dalam dokumen (opsional)
    "Teks_Soal",        # Teks instrumen soal yang diklasifikasikan
    "KKO",              # Kata Kerja Operasional yang diekstrak
    "Level_Bloom",      # Hasil klasifikasi (mis. "C4 - Menganalisis")
    "Kategori_HOTS_LOTS",# "HOTS" atau "LOTS"
    "Model_Used",       # Model AI yang digunakan ("gemini-1.5-flash" atau "mistral")
    "Confidence_Note",  # Catatan kepercayaan/alasan dari model (opsional)
]


# =============================================================================
# BAGIAN 2: SETUP LOGURU (Log Teks)
# =============================================================================

def setup_text_logger(
    log_file: Path = LOG_FILE_PATH,
    log_level: str = LOG_LEVEL,
) -> None:
    """
    Mengonfigurasi logger teks (loguru) dengan output ke konsol dan file.

    Args:
        log_file: Path ke file log teks yang akan ditulis.
        log_level: Level minimum log yang dicatat (DEBUG/INFO/WARNING/ERROR).
    """
    # Hapus semua handler default loguru untuk konfigurasi ulang dari awal
    _loguru_logger.remove()

    # Handler 1: Output ke konsol (stdout) dengan warna
    _loguru_logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Handler 2: Output ke file dengan rotasi otomatis (maks 10 MB per file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _loguru_logger.add(
        str(log_file),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",   # Buat file baru setiap log mencapai 10 MB
        retention="30 days",# Hapus log lama setelah 30 hari
        encoding="utf-8",
    )

    _loguru_logger.info(f"Logger teks diinisialisasi. File log: {log_file}")


# Ekspor `logger` agar bisa diimpor langsung: `from utils.logger import logger`
logger = _loguru_logger


# =============================================================================
# BAGIAN 3: CLASS CSVClassificationLogger (Log CSV Terstruktur)
# =============================================================================

class CSVClassificationLogger:
    """
    Mencatat hasil klasifikasi Taksonomi Bloom ke dalam file CSV secara otomatis.

    Setiap kali `log_result()` dipanggil, satu baris baru akan ditambahkan
    ke file CSV tanpa menghapus data sebelumnya (mode append).

    Penggunaan:
        csv_logger = CSVClassificationLogger()
        csv_logger.log_result(
            nama_file="soal_ujian.pdf",
            teks_soal="Analisislah dampak dari...",
            kko="Analisislah",
            level_bloom="C4 - Menganalisis",
            model_used="gemini-1.5-flash",
        )
    """

    def __init__(self, csv_path: Path = LOG_CSV_PATH) -> None:
        """
        Inisialisasi logger CSV.

        Args:
            csv_path: Path ke file CSV tujuan. File akan dibuat otomatis
                      beserta direktori induknya jika belum ada.
        """
        self.csv_path: Path = csv_path
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """
        Membuat file CSV dengan baris header jika file belum ada.
        Jika file sudah ada, tidak ada perubahan (data lama aman).
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
                # utf-8-sig agar file CSV terbuka dengan benar di Microsoft Excel
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info(f"File CSV baru dibuat: {self.csv_path}")
        else:
            logger.info(f"File CSV sudah ada, data akan di-append: {self.csv_path}")

    def log_result(
        self,
        teks_soal: str,
        kko: str,
        level_bloom: str,
        model_used: str,
        nama_file: str = "unknown.pdf",
        nomor_soal: Optional[int] = None,
        confidence_note: Optional[str] = None,
    ) -> None:
        """
        Mencatat SATU hasil klasifikasi ke dalam file CSV.

        Args:
            teks_soal:       Teks soal yang diklasifikasikan.
            kko:             Kata Kerja Operasional yang ditemukan.
            level_bloom:     Level Bloom hasil klasifikasi (mis. "C4 - Menganalisis").
            model_used:      Nama model AI yang menghasilkan klasifikasi.
            nama_file:       Nama file PDF sumber soal.
            nomor_soal:      Nomor urut soal (opsional).
            confidence_note: Catatan alasan atau kepercayaan dari model (opsional).
        """
        # Tentukan kategori HOTS atau LOTS berdasarkan level
        hots_indicators: list[str] = ["C4", "C5", "C6"]
        kategori: str = "HOTS" if any(
            level_bloom.startswith(h) for h in hots_indicators
        ) else "LOTS"

        # Buat baris data sesuai urutan kolom CSV
        row: dict[str, str] = {
            "Timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nama_File":          nama_file,
            "Nomor_Soal":         str(nomor_soal) if nomor_soal is not None else "-",
            "Teks_Soal":          teks_soal.strip(),
            "KKO":                kko.strip(),
            "Level_Bloom":        level_bloom.strip(),
            "Kategori_HOTS_LOTS": kategori,
            "Model_Used":         model_used.strip(),
            "Confidence_Note":    (confidence_note or "-").strip(),
        }

        # Tulis ke CSV dalam mode append
        with open(self.csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        logger.info(
            f"[CSV LOG] {nama_file} | Soal #{nomor_soal} | "
            f"KKO: '{kko}' | {level_bloom} ({kategori}) | Model: {model_used}"
        )

    def log_batch(self, results: list[dict]) -> None:
        """
        Mencatat BANYAK hasil klasifikasi sekaligus (lebih efisien untuk batch).

        Args:
            results: List of dict, setiap dict adalah argumen untuk `log_result()`.
                     Kunci yang diwajibkan: teks_soal, kko, level_bloom, model_used.
        """
        logger.info(f"Memulai batch logging untuk {len(results)} soal...")
        for i, result in enumerate(results, start=1):
            try:
                self.log_result(**result)
            except (KeyError, TypeError) as e:
                logger.error(f"Gagal mencatat soal ke-{i} dalam batch: {e}")
        logger.success(f"Batch logging selesai. {len(results)} baris ditulis ke CSV.")

    def get_summary(self) -> dict[str, int]:
        """
        Membaca file CSV dan mengembalikan ringkasan jumlah soal per level Bloom.

        Returns:
            Dictionary dengan level Bloom sebagai kunci dan jumlah soal sebagai nilai.
        """
        if not self.csv_path.exists():
            logger.warning("File CSV belum ada. Belum ada data yang bisa dirangkum.")
            return {}

        summary: dict[str, int] = {}
        try:
            with open(self.csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    level = row.get("Level_Bloom", "Tidak Diketahui")
                    summary[level] = summary.get(level, 0) + 1
        except Exception as e:
            logger.error(f"Gagal membaca ringkasan CSV: {e}")

        return summary

    def __repr__(self) -> str:
        return f"CSVClassificationLogger(csv_path='{self.csv_path}')"


# =============================================================================
# BAGIAN 4: FUNGSI UTILITAS TINGKAT ATAS (Top-level Helper Functions)
# =============================================================================

def get_csv_logger(csv_path: Optional[Path] = None) -> CSVClassificationLogger:
    """
    Factory function untuk mendapatkan instance CSVClassificationLogger.
    Memudahkan penggunaan dengan path kustom tanpa perlu import class secara langsung.

    Args:
        csv_path: Path opsional ke file CSV. Jika None, gunakan path dari config.

    Returns:
        Instance CSVClassificationLogger yang siap digunakan.
    """
    path = csv_path or LOG_CSV_PATH
    return CSVClassificationLogger(csv_path=path)


# =============================================================================
# BAGIAN 5: INISIALISASI OTOMATIS SAAT MODUL DIIMPOR
# =============================================================================

# Setup text logger dilakukan sekali saat modul pertama kali diimpor
setup_text_logger()

# Instance CSV logger default yang siap pakai
# Contoh penggunaan: from utils.logger import csv_logger
csv_logger: CSVClassificationLogger = CSVClassificationLogger()


# =============================================================================
# CONTOH PENGGUNAAN (untuk testing manual: python utils/logger.py)
# =============================================================================
if __name__ == "__main__":
    logger.info("=== Menjalankan test mandiri utils/logger.py ===")

    # Contoh log satu soal
    csv_logger.log_result(
        nama_file="contoh_soal_ujian.pdf",
        nomor_soal=1,
        teks_soal="Analisislah hubungan antara perubahan iklim dan ketahanan pangan di Indonesia.",
        kko="Analisislah",
        level_bloom="C4 - Menganalisis",
        model_used="gemini-1.5-flash",
        confidence_note="KKO 'Analisislah' secara eksplisit menunjuk ke domain C4.",
    )

    csv_logger.log_result(
        nama_file="contoh_soal_ujian.pdf",
        nomor_soal=2,
        teks_soal="Sebutkan tiga faktor yang mempengaruhi pertumbuhan tanaman.",
        kko="Sebutkan",
        level_bloom="C1 - Mengingat",
        model_used="mistral",
        confidence_note="Fallback ke model lokal karena timeout API Gemini.",
    )

    # Tampilkan ringkasan
    summary = csv_logger.get_summary()
    logger.info(f"Ringkasan hasil klasifikasi: {summary}")
    logger.success(f"File CSV tersimpan di: {csv_logger.csv_path}")
