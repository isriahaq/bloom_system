# =============================================================================
# modules/llm_router.py
# Router AI Cloud untuk Klasifikasi Taksonomi Bloom
# =============================================================================
# Modul ini merupakan inti pemrosesan kognitif dari sistem klasifikasi soal.
# Ia berfungsi sebagai "Router Cerdas" yang menerapkan arsitektur kaskade
# (Cascading Models) secara murni berbasis cloud Google Gemini API.
#
# Alur Arsitektur (Cascading Fallback antar-Varian Gemini):
#   evaluate_batch(questions)
#      └─► _call_gemini_cascade()  ← SATU-SATUNYA rute (Pure Cloud)
#              ├─ Mencoba 'gemini-3.7-flash'   (Prioritas 1 — tertinggi)
#              ├─ [Jika 429/5xx/timeout] ─► 'gemini-3.6-flash' (Prioritas 2)
#              └─ [Jika masih gagal]    ─► 'gemini-3.5-flash'  (Prioritas 3)
#
# Jika seluruh rantai model gagal, sistem mencatat error dan mengembalikan
# placeholder kosong agar pipeline tidak crash.
#
# Seluruh output LLM dipaksa secara deterministik menggunakan skema
# Pydantic (Structured Outputs) guna menjamin tidak adanya deviasi parsing JSON.
# =============================================================================

import json
import re
import time
from typing import Optional

from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types

# =============================================================================
# KONFIGURASI DAN DEPENDENSI EKSTERNAL
# =============================================================================
try:
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODELS,
        GEMINI_MAX_RETRIES,
    )
    from utils.logger import logger
except ImportError:
    # Fallback pencegahan agar modul tetap bisa diuji terpisah tanpa crash
    import logging
    import os
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # type: ignore[assignment]
    GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODELS     = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-2.5-flash")
    GEMINI_MAX_RETRIES = 3


# =============================================================================
# SKEMA STRUKTUR DATA (PYDANTIC)
# =============================================================================
class EvaluasiSoal(BaseModel):
    """Skema dasar entitas hasil klasifikasi satu soal."""
    question_text: str
    is_question: bool
    tipe_soal: Optional[str] = None
    kko: Optional[str] = None
    level: Optional[str] = None
    kategori: Optional[str] = None
    anomali_opsi: Optional[str] = None

class HasilBatch(BaseModel):
    """Skema pembungkus (wrapper) untuk respons kumpulan soal dari Gemini."""
    evaluasi_soal: list[EvaluasiSoal]


# =============================================================================
# KONSTANTA SISTEM
# =============================================================================
_VALID_LEVELS: frozenset[str] = frozenset({"C1", "C2", "C3", "C4", "C5", "C6"})
_VALID_CATEGORIES: frozenset[str] = frozenset({"LOTS", "MOTS", "HOTS"})


# =============================================================================
# SYSTEM PROMPT
# =============================================================================
SYSTEM_PROMPT_BATCH: str = """Anda adalah sistem analisis kognitif berbasis Taksonomi Bloom untuk soal-soal pemrograman dasar. Tugas utama Anda adalah melakukan klasifikasi secara akurat.

# ATURAN PRESERVASI TEKS (SANGAT KETAT):
1. **Dilarang Keras Memotong/Meringkas Teks (No Truncation)**: Tulis kembali soal beserta semua pilihannya secara 100% UTUH pada field "question_text".
2. **Format Baris Baru**: Gunakan karakter escape "\\n". DILARANG menggunakan tag HTML.
3. **Noise Filtering**: Jika input BUKAN soal, set "is_question": false, null-kan kko, level, kategori. Teks tetap di-preserve.

TUGAS UTAMA:
1. Klasifikasi Bloom: Identifikasi Kata Kerja Operasional (KKO) utama, Level (C1-C6), dan Kategori (LOTS/MOTS/HOTS).
2. Tipe Soal: Tentukan apakah soal ini "Pilihan Ganda" atau "Esai".
3. Deteksi Anomali Opsi: KHUSUS soal "Pilihan Ganda", bandingkan opsi (A, B, C, D, E). Jika ada opsi ganda/identik, laporkan. Jika soal "Esai", isi field ini dengan tulisan "Tidak relevan (Esai)", jangan isi "Aman" atau "None".

PANDUAN KLASIFIKASI:
- C1 -> LOTS
- C2, C3 -> MOTS
- C4, C5, C6 -> HOTS

# CONTOH FEW-SHOT PROMPTING
Berikut adalah contoh format output yang diharapkan (Format JSON akan menyesuaikan skema Pydantic):

Contoh 1 (Normal, Tanpa Anomali):
Input: "[1] Manakah dari berikut ini yang merupakan tipe data untuk teks? A. int B. float C. string D. boolean E. array"
Output (Konseptual):
{
  "question_text": "Manakah dari berikut ini yang merupakan tipe data untuk teks?\\nA. int\\nB. float\\nC. string\\nD. boolean\\nE. array",
  "is_question": true,
  "tipe_soal": "Pilihan Ganda",
  "kko": "pilih",
  "level": "C1",
  "kategori": "LOTS",
  "anomali_opsi": "Aman"
}

Contoh 2 (Mengandung Anomali Duplikasi Opsi):
Input: "[2] Apa keyword untuk perulangan di Python? A. for B. while C. loop D. for E. do-while"
Output (Konseptual):
{
  "question_text": "Apa keyword untuk perulangan di Python?\\nA. for\\nB. while\\nC. loop\\nD. for\\nE. do-while",
  "is_question": true,
  "tipe_soal": "Pilihan Ganda",
  "kko": "identifikasi",
  "level": "C1",
  "kategori": "LOTS",
  "anomali_opsi": "Terdapat duplikasi pada opsi A dan D"
}

Contoh 3 (Soal Esai):
Input: "[3] Jelaskan perbedaan antara perulangan for dan while dalam pemrograman!"
Output (Konseptual):
{
  "question_text": "Jelaskan perbedaan antara perulangan for dan while dalam pemrograman!",
  "is_question": true,
  "tipe_soal": "Esai",
  "kko": "jelaskan",
  "level": "C2",
  "kategori": "MOTS",
  "anomali_opsi": "Tidak relevan (Esai)"
}

Periksa dengan cermat setiap opsi sebelum mengisi kolom anomali_opsi!
"""


# =============================================================================
# KELAS INTI ROUTER LLM (Pure Cloud — Cascading Gemini)
# =============================================================================
class CloudLLMRouter:
    """
    Router AI berbasis Google Gemini dengan arsitektur Cascading Fallback.

    Kelas ini memfasilitasi komunikasi menuju Google Generative AI (Gemini)
    menggunakan strategi kaskade prioritas model. Sistem akan mencoba
    model secara berurutan (gemini-2.5-flash → gemini-2.0-flash → gemini-1.5-flash).
    
    Jika satu model mengalami error (rate limit 429, server error 5xx, timeout,
    atau connection error), beban payload secara otomatis berpindah ke model
    berikutnya dalam rantai GEMINI_MODELS tanpa menghentikan aplikasi.
    """

    def __init__(
        self,
        gemini_api_key: str = GEMINI_API_KEY,
        gemini_max_retries: int = GEMINI_MAX_RETRIES,
    ) -> None:
        self.gemini_max_retries = gemini_max_retries
        self._gemini_client: Optional[genai.Client] = None
        self._active_model: Optional[str] = None  # Untuk tracking model yang sedang aktif

        if gemini_api_key:
            try:
                self._gemini_client = genai.Client(api_key=gemini_api_key)
                logger.info("CloudLLMRouter: Klien Google GenAI berhasil terinisialisasi.")
            except Exception as e:
                logger.error(f"CloudLLMRouter: Gagal memuat API Key ({e}).")
        else:
            logger.warning("CloudLLMRouter: Tidak ada API Key Gemini yang terdeteksi.")

    @property
    def active_model(self) -> Optional[str]:
        """Model Gemini yang berhasil merespons pada batch terakhir."""
        return self._active_model

    def evaluate_batch(self, questions: list[str], batch_size: int = 10) -> list[dict]:
        """
        Titik Masuk Utama (Entry Point) untuk evaluasi soal dalam jumlah besar.

        Fungsi ini melakukan penyekatan (chunking) otomatis pada daftar soal
        agar masuk ke dalam batas toleransi model, lalu memerintahkan eksekusi
        evaluasi berbasis kelompok (batch) dengan strategi cascading antar-model Gemini.

        Args:
            questions (list[str]): Deret naskah soal mentah dari PDF.
            batch_size (int): Kapasitas soal yang dimuat dalam 1 kali request API.

        Returns:
            list[dict]: Barisan hasil yang memiliki key eksak KKO, Level, dan Kategori.
        """
        if not questions:
            return []

        results: list[dict] = []
        total = len(questions)

        for i in range(0, total, batch_size):
            chunk = questions[i:i + batch_size]

            # Eksekusi kaskade Gemini untuk batch ini
            chunk_results = self._call_gemini_cascade(chunk)
            results.extend(chunk_results)

            # Penjedaan rate limiting (anti spam) antar batch
            if i + batch_size < total:
                time.sleep(3)

        return results

    def _call_gemini_cascade(self, questions_chunk: list[str]) -> list[dict]:
        """
        Mengeksekusi klasifikasi soal menggunakan strategi Cascading Fallback
        secara eksklusif di dalam ekosistem Gemini API.

        Alur:
        1. Iterasi dimulai dari model prioritas pertama dalam GEMINI_MODELS.
        2. Jika terjadi Rate Limit (429) atau Server Error (5xx), sistem langsung
           melakukan short-circuit dan berpindah ke model berikutnya.
        3. Jika terjadi error koneksi/timeout pada model yang sama, sistem
           melakukan retry hingga GEMINI_MAX_RETRIES kali dengan exponential backoff.
        4. Jika seluruh rantai habis, mengembalikan placeholder error.

        Args:
            questions_chunk: Potongan daftar soal untuk diproses dalam satu request.

        Returns:
            list[dict]: Hasil klasifikasi atau daftar error placeholder.
        """
        if not self._gemini_client:
            return [{"error": "Klien API tidak aktif (API Key tidak valid atau kosong)"}
                    for _ in questions_chunk]

        numbered_questions = "\n".join(
            f"[{i + 1}] {q.strip()}" for i, q in enumerate(questions_chunk)
        )
        batch_prompt = (
            f"{SYSTEM_PROMPT_BATCH}\n\n---\n"
            f"Klasifikasikan soal-soal berikut:\n"
            f"{numbered_questions}"
        )

        last_error = "Belum dieksekusi."

        # Iterasi rantai model kaskade
        for model_name in GEMINI_MODELS:
            logger.info(f"[Gemini-Cascading] Berusaha merutekan ke: {model_name}")

            for attempt in range(1, self.gemini_max_retries + 1):
                try:
                    generate_config = genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT_BATCH,
                        response_mime_type="application/json",
                        response_schema=HasilBatch,
                        temperature=0.1,
                    )

                    chat = self._gemini_client.chats.create(
                        model=model_name,
                        config=generate_config,
                    )
                    response = chat.send_message(batch_prompt)

                    raw_text = response.text if response.text else ""
                    parsed = self._parse_batch_response(raw_text, len(questions_chunk), model_name)

                    if parsed:
                        self._active_model = model_name
                        for p in parsed:
                            p["model_used"] = model_name
                        logger.info(
                            f"[Gemini-Cascading] ✓ Berhasil via {model_name} "
                            f"({len(parsed)} hasil)"
                        )
                        return parsed

                    last_error = f"Parsing Ditolak ({model_name})"

                except Exception as e:
                    str_e = str(e).lower()

                    # Deteksi jenis error untuk menentukan strategi penanganan
                    is_rate_limited  = any(kw in str_e for kw in ("quota", "429", "rate", "exhausted"))
                    is_not_found     = any(kw in str_e for kw in ("404", "not found"))
                    is_server_error  = any(kw in str_e for kw in (
                        "503", "500", "502", "504", "unavailable", "overloaded"
                    ))
                    is_connect_error = any(kw in str_e for kw in (
                        "connect", "timeout", "network", "getaddrinfo", "errno"
                    ))

                    if is_rate_limited:
                        logger.warning(
                            f"[Gemini-Cascading] Rate Limit (429) pada {model_name}. "
                            f"Memindahkan ke model berikutnya..."
                        )
                        last_error = f"Rate Limit: {model_name}"
                        break  # Short-circuit: langsung ke model berikutnya

                    elif is_not_found:
                        logger.warning(
                            f"[Gemini-Cascading] Model tidak ditemukan (404): {model_name}. "
                            f"Memindahkan ke model berikutnya..."
                        )
                        last_error = f"Model tidak tersedia: {model_name}"
                        break  # Short-circuit: langsung ke model berikutnya

                    elif is_server_error or is_connect_error:
                        last_error = f"{type(e).__name__}: {e}"
                        logger.error(
                            f"[Gemini-Cascading] Gangguan Server/Koneksi pada {model_name} "
                            f"(Percobaan {attempt}/{self.gemini_max_retries}): {last_error}"
                        )
                        if attempt < self.gemini_max_retries:
                            wait_secs = 2 ** attempt
                            logger.info(f"[Gemini-Cascading] Retry dalam {wait_secs} detik...")
                            time.sleep(wait_secs)
                        else:
                            logger.warning(
                                f"[Gemini-Cascading] Percobaan habis pada {model_name}. "
                                f"Memindahkan ke model berikutnya..."
                            )
                            break  # Retry habis: pindah ke model berikutnya

                    else:
                        last_error = f"{type(e).__name__}: {e}"
                        logger.error(
                            f"[Gemini-Cascading] Error tak terduga pada {model_name} "
                            f"(Percobaan {attempt}): {last_error}"
                        )
                        if attempt < self.gemini_max_retries:
                            time.sleep(2 ** attempt)
                        else:
                            break

        # Seluruh rantai kaskade habis tanpa hasil
        self._active_model = None
        logger.error(
            f"[Gemini-Cascading] Seluruh rantai model gagal. "
            f"Error terakhir: {last_error}"
        )
        return [{"error": f"Kaskade Penuh Gagal: {last_error}"} for _ in questions_chunk]

    def _parse_batch_response(
        self, raw_text: str, expected: int, source: str
    ) -> Optional[list[dict]]:
        """
        Parser dan Auto-Linter JSON Robust.

        Mencoba meluruskan penyimpangan pada output LLM dengan sistem ekstraksi
        kamus bebas huruf besar (case-insensitive keys) dan RegExp failsafe
        (Pattern Matching).

        Args:
            raw_text: Teks mentah respons dari LLM.
            expected: Jumlah soal yang diharapkan dalam batch ini.
            source: Nama model/sumber untuk log.

        Returns:
            list[dict] hasil yang sudah dinormalisasi, atau None jika parsing gagal.
        """
        if not raw_text or not raw_text.strip():
            return None

        cleaned = raw_text.strip()

        # Bersihkan markdown code block jika ada
        md_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", cleaned, re.DOTALL)
        if md_match:
            cleaned = md_match.group(1)

        # Ekstrak hanya teks JSON (array atau objek) dari respons
        sanitizer_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if sanitizer_match:
            cleaned = sanitizer_match.group(1)

        try:
            data_raw = json.loads(cleaned, strict=False)
            # Normalisasi respons Pydantic wrapper (GenerativeModel response_schema behavior)
            if isinstance(data_raw, dict):
                if "evaluasi_soal" in data_raw:
                    data = data_raw["evaluasi_soal"]
                elif "evaluasi" in data_raw:
                    data = data_raw["evaluasi"]
                else:
                    data = data_raw
            else:
                data = data_raw

        except json.JSONDecodeError as e:
            # Toleransi untuk output JSONL (multiple objects tanpa array pembungkus)
            if "Extra data" in str(e):
                try:
                    fixed_cleaned = re.sub(r'\}\s*\{', '},{', cleaned)
                    fixed_cleaned = f"[{fixed_cleaned}]"
                    data_raw = json.loads(fixed_cleaned, strict=False)
                    data = data_raw
                except Exception as inner_e:
                    logger.error(
                        f"[{source}] JSON Korup dan auto-fix gagal. "
                        f"{e} | {inner_e}"
                    )
                    return None
            else:
                logger.warning(f"[{source}] JSON Korup. Mencoba pemulihan darurat (Auto-Fix)...")
                try:
                    import json_repair
                    data_raw = json_repair.loads(cleaned)
                    data = data_raw
                except ImportError:
                    # Fallback manual jika modul json_repair tidak terinstal
                    temp_cleaned = cleaned.rstrip()
                    if temp_cleaned.count('"') % 2 != 0:
                        temp_cleaned += '"'
                    if not temp_cleaned.endswith("}"):
                        temp_cleaned += "}"
                    if not temp_cleaned.endswith("]"):
                        temp_cleaned += "]"
                    try:
                        data_raw = json.loads(temp_cleaned, strict=False)
                        data = data_raw
                    except json.JSONDecodeError as final_e:
                        logger.error(f"[{source}] Pemulihan darurat gagal. {final_e}")
                        return None

        if not isinstance(data, list) or len(data) == 0:
            return None

        validated = []
        for item in data:
            if not isinstance(item, dict):
                continue

            normalized = {str(k).lower().strip(): v for k, v in item.items()}

            is_question = normalized.get("is_question", True)
            if isinstance(is_question, str):
                is_question = str(is_question).lower() == "true"

            question_text = str(
                normalized.get("question_text", normalized.get("butir_soal", ""))
            )
            kko = (
                str(normalized.get("kko", "")).strip().lower()
                if normalized.get("kko")
                else None
            )

            # Override: Jika question_text berisi teks soal nyata, anggap is_question=True.
            # Teks PDF asli sudah di-inject dari app.py, jadi LLM tidak bisa
            # menghilangkan soal hanya karena salah membaca konteks batch-nya.
            if question_text and len(question_text.strip()) > 15:
                is_question = True

            level = "N/A"
            kategori = "N/A"

            if is_question and kko and kko not in ("none", "null"):
                # Level Toleransi Auto-Linting
                level_raw = str(normalized.get("level", "")).strip().upper()
                if not level_raw or level_raw not in _VALID_LEVELS:
                    match_level = re.search(r"C[1-6]", str(item).upper())
                    level = match_level.group(0) if match_level else "C0"
                else:
                    level = level_raw

                # Kategori Toleransi Auto-Linting
                kategori_raw = str(
                    normalized.get("kategori")
                    or normalized.get("category")
                    or normalized.get("kategori_soal", "")
                ).strip().upper()

                if not kategori_raw or kategori_raw not in _VALID_CATEGORIES:
                    match_cat = re.search(r"\b(LOTS|MOTS|HOTS)\b", str(item).upper())
                    kategori = match_cat.group(1) if match_cat else "ERROR"
                else:
                    kategori = kategori_raw

                # Fallback Kategori: auto-infer dari level jika masih ERROR
                if kategori == "ERROR" and level in _VALID_LEVELS:
                    if level == "C1":
                        kategori = "LOTS"
                    elif level in ("C2", "C3"):
                        kategori = "MOTS"
                    elif level in ("C4", "C5", "C6"):
                        kategori = "HOTS"

            elif is_question:
                # KKO tidak ada tapi soal valid — tetap ekstrak level dan kategori
                kko_raw = str(normalized.get("kko", "")).strip().lower()
                kko = kko_raw if kko_raw and kko_raw not in ("none", "null", "n/a", "") else "N/A"

                level_raw = str(normalized.get("level", "")).strip().upper()
                level = level_raw if level_raw in _VALID_LEVELS else "N/A"

                kategori_raw = str(
                    normalized.get("kategori") or normalized.get("category", "")
                ).strip().upper()

                if kategori_raw in _VALID_CATEGORIES:
                    kategori = kategori_raw
                elif level in _VALID_LEVELS:
                    if level == "C1":
                        kategori = "LOTS"
                    elif level in ("C2", "C3"):
                        kategori = "MOTS"
                    elif level in ("C4", "C5", "C6"):
                        kategori = "HOTS"
            else:
                kko = "N/A"

            tipe_soal = str(normalized.get("tipe_soal", "Pilihan Ganda")).strip()
            anomali_opsi = str(normalized.get("anomali_opsi", "Aman")).strip()

            # Paksa anomali_opsi menjadi "Tidak relevan (Esai)" jika tipe soalnya Esai
            if tipe_soal.lower() == "esai":
                anomali_opsi = "Tidak relevan (Esai)"

            validated.append({
                "question_text": question_text,
                "is_question": is_question,
                "tipe_soal": tipe_soal,
                "kko": kko,
                "level": level,
                "kategori": kategori,
                "anomali_opsi": anomali_opsi,
            })

        if not validated:
            return None

        return validated


# =============================================================================
# ALIAS KOMPATIBILITAS MUNDUR (Backward Compatibility)
# =============================================================================
# Alias ini memastikan kode di app.py yang masih menggunakan nama "HybridLLMRouter"
# tidak perlu diubah, cukup merujuk ke kelas baru CloudLLMRouter.
HybridLLMRouter = CloudLLMRouter
