# =============================================================================
# modules/pdf_processor.py
# Modul Pemrosesan Dokumen PDF untuk Sistem Hybrid LLM Taksonomi Bloom
# =============================================================================
# Tanggung Jawab Modul Ini:
#   1. Membuka dan membaca file PDF menggunakan PyMuPDF (fitz)
#   2. Mengekstrak teks mentah dari seluruh halaman
#   3. Membersihkan residu dokumen (header, footer, nomor halaman, dll.)
#   4. Memisahkan (segmentasi) teks menjadi butir-butir soal individual
#
# CATATAN KRUSIAL — Aturan Pilihan Ganda:
#   Teks opsi jawaban (A., B., C., D., E.) harus SELALU tetap menyatu
#   dengan soal utamanya dalam satu elemen list. Opsi jawaban tidak boleh
#   terpotong menjadi elemen list yang terpisah dari soal pokoknya.
#
# Fokus Dokumen:
#   Modul ini dirancang khusus untuk dokumen Instrumen Soal/Ujian dengan
#   format campuran: pilihan ganda (A–E) dan soal esai.
# =============================================================================

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — pastikan sudah: pip install PyMuPDF

# Gunakan logger terpusat dari utils; fallback ke logging standar jika dijalankan mandiri
try:
    from utils.logger import logger
    from config import INPUT_DIR, OUTPUT_DIR
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # type: ignore[assignment]
    INPUT_DIR = Path("data/input")
    OUTPUT_DIR = Path("data/output")


# =============================================================================
# BAGIAN 1: DATACLASS HASIL PEMROSESAN
# =============================================================================

@dataclass
class ProcessedDocument:
    """
    Data container yang merepresentasikan hasil pemrosesan satu dokumen PDF.

    Atribut:
        file_path:      Path absolut ke file PDF sumber.
        file_name:      Nama file PDF (basename saja).
        total_pages:    Jumlah halaman yang berhasil dibaca.
        raw_text:       Teks mentah gabungan dari seluruh halaman.
        questions:      List butir soal yang sudah bersih dan tersegmentasi.
        error_message:  Pesan error jika pemrosesan gagal (None jika sukses).
    """
    file_path: str
    file_name: str
    total_pages: int = 0
    raw_text: str = ""
    questions: list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Mengembalikan True jika dokumen berhasil diproses tanpa error."""
        return self.error_message is None

    @property
    def question_count(self) -> int:
        """Mengembalikan jumlah butir soal yang berhasil diekstrak."""
        return len(self.questions)

    def __repr__(self) -> str:
        status = "OK" if self.is_success else f"ERROR: {self.error_message}"
        return (
            f"ProcessedDocument("
            f"file='{self.file_name}', "
            f"pages={self.total_pages}, "
            f"questions={self.question_count}, "
            f"status='{status}')"
        )


# =============================================================================
# BAGIAN 2: POLA REGEX — DIKOMPILASI SEKALI SEBAGAI KONSTANTA MODUL
# =============================================================================
# Semua pola regex didefinisikan sebagai konstanta modul agar dikompilasi
# hanya satu kali saat modul pertama kali diimpor (lebih efisien).

# --- Regex 1: Penanda batas halaman internal (ditambahkan oleh extract_raw_text) ---
# Menghapus baris: "--- [HALAMAN 1] ---", "--- [HALAMAN 12] ---", dst.
_RE_PAGE_MARKER: re.Pattern = re.compile(
    r"---\s*\[HALAMAN\s*\d+\]\s*---"
)

# --- Regex 2: Nomor halaman dari dokumen asli ---
# Menangani: "Hal 1 dari 5", "Halaman 2", "Page 3 of 10", "- 4 -"
_RE_PAGE_NUMBER: re.Pattern = re.compile(
    r"(?i)(hal(?:aman)?\s*\d+\s*(?:dari\s*\d+)?|page\s*\d+\s*(?:of\s*\d+)?|\-\s*\d+\s*\-)",
    re.IGNORECASE,
)

# --- Regex 3: Karakter non-printable dan karakter aneh hasil OCR ---
# Mempertahankan huruf Latin beraksent (é, ñ, ä, dst.) dan karakter whitespace normal.
_RE_NON_PRINTABLE: re.Pattern = re.compile(
    r"[^\x20-\x7E\u00C0-\u024F\u0100-\u017F\n\r\t]"
)

# --- Regex 4: Baris sisa header/footer yang hanya berisi angka saja ---
# PENTING: Pattern ini HANYA mencocokkan baris berisi ANGKA murni (bukan huruf).
# Ini sengaja dibatasi agar tidak ikut menghapus baris opsi jawaban seperti "A. Pilihan".
# Pola asli [\d\W]+ terlalu agresif — bisa salah menghapus baris opsi A, B, C, D, E.
_RE_LONE_NUMBER_LINE: re.Pattern = re.compile(
    r"^\s*\d+\s*$", re.MULTILINE
)

# --- Regex 5: Spasi berlebihan dalam satu baris ---
_RE_EXCESS_SPACES: re.Pattern = re.compile(r"[ \t]{2,}")

# --- Regex 6: Baris kosong berlebihan (lebih dari 2 berturut-turut) ---
_RE_EXCESS_BLANK_LINES: re.Pattern = re.compile(r"\n{3,}")

# --- Regex 7: DELIMITER SOAL — Inti logika segmentasi ---
# Pola ini mengenali AWAL dari butir soal baru. Menggunakan LOOKAHEAD (?=...)
# agar pemisah (nomor soal) ikut tersimpan di awal segmen, tidak terhapus saat split.
#
# ATURAN KRUSIAL PILIHAN GANDA:
# Regex ini secara SENGAJA hanya mencocokkan nomor angka (1–150), BUKAN huruf (A–E).
# Hal ini memastikan opsi jawaban "A. ...", "B. ...", dst. TIDAK dianggap sebagai
# pemisah soal baru, sehingga tetap menyatu dengan teks pertanyaan utamanya.
#
# Format yang didukung:
#   "1. Teks soal..."   - nomor lalu titik lalu spasi
#   "1) Teks soal..."   - nomor lalu kurung tutup lalu spasi
#   "No. 1 Teks soal..." - awalan "No." diikuti nomor
#   "Soal 1. Teks soal..." - awalan "Soal" diikuti nomor
#
# TIDAK akan cocok dengan:
#   "A. Pilihan jawaban" - karena A bukan angka
#   "B) Pilihan jawaban" - karena B bukan angka
#   "10" sendiri di akhir baris (tidak diikuti karakter non-spasi)
_RE_QUESTION_DELIMITER: re.Pattern = re.compile(
    r"^(?=[ \t]*(?:(?:No\.?|NO\.?|no\.?|Soal|SOAL|soal|Question|QUESTION|question|Pertanyaan|PERTANYAAN|pertanyaan|Q)\s*)?\d{1,3}(?:\.|\)|\]|\s+-\s+|\s+(?=[A-Z])|\s*\n))",
    re.MULTILINE,
)

# --- Regex 8: Validasi apakah segmen adalah soal valid ---
# Segmen dianggap soal valid jika diawali pola nomor soal
_RE_VALID_QUESTION_START: re.Pattern = re.compile(
    r"^[ \t]*(?:(?:No\.?|NO\.?|no\.?|Soal|SOAL|soal|Question|QUESTION|question|Pertanyaan|PERTANYAAN|pertanyaan|Q)\s*)?\d{1,3}(?:\.|\)|\]|\s+-\s+|\s+(?=[A-Z])|\s*\n)",
    re.MULTILINE,
)


# =============================================================================
# BAGIAN 3: CLASS UTAMA PDFProcessor
# =============================================================================

class PDFProcessor:
    """
    Kelas utama untuk memproses dokumen PDF instrumen asesmen (soal ujian).

    Fokus dokumen yang didukung:
        - Soal pilihan ganda format A/B/C/D/E (opsi tetap menyatu dengan soal)
        - Soal esai / uraian singkat

    Alur Kerja Internal:
        PDF File → extract_raw_text() → clean_and_segment_text() → list[str] soal

    Contoh Penggunaan Sederhana:
        processor = PDFProcessor()
        questions = processor.process_document("data/input/soal_ujian.pdf")
        for soal in questions:
            print(soal)

    Contoh Penggunaan Lengkap (dengan metadata):
        processor = PDFProcessor()
        result = processor.process_document_full("data/input/soal_ujian.pdf")
        print(f"Berhasil: {result.is_success}")
        print(f"Jumlah soal: {result.question_count}")
        print(f"Soal pertama: {result.questions[0]}")
    """

    def __init__(
        self,
        min_question_length: int = 10,
        strip_lone_numbers: bool = True,
    ) -> None:
        """
        Inisialisasi PDFProcessor.

        Args:
            min_question_length: Panjang karakter minimum sebuah string dianggap
                                 sebagai soal valid (filter noise pendek).
            strip_lone_numbers:  Jika True, hapus baris yang hanya berisi angka
                                 saja (sisa penomoran halaman dari header/footer).
                                 Berbeda dari versi sebelumnya: TIDAK menghapus
                                 semua simbol agar opsi A–E aman.
        """
        self.min_question_length: int = min_question_length
        self.strip_lone_numbers: bool = strip_lone_numbers
        logger.debug(
            f"PDFProcessor diinisialisasi "
            f"(min_q_len={min_question_length}, strip_lone_numbers={strip_lone_numbers})"
        )

    # -------------------------------------------------------------------------
    # METHOD 1: extract_raw_text
    # -------------------------------------------------------------------------

    def extract_raw_text(self, file_path: str) -> str:
        """
        Mengekstrak teks mentah (raw) dari seluruh halaman dokumen PDF.

        Membuka file PDF menggunakan PyMuPDF (fitz), iterasi setiap halaman,
        lalu menggabungkan seluruh teks dengan penanda batas halaman internal.
        Penanda halaman ini berguna untuk debugging dan akan dihapus pada
        tahap clean_and_segment_text().

        Args:
            file_path: Path ke file PDF sumber (absolut atau relatif terhadap CWD).

        Returns:
            String teks mentah gabungan dari semua halaman, dipisahkan oleh
            penanda "--- [HALAMAN X] ---".
            Mengembalikan string kosong "" jika tidak ada teks yang berhasil
            diekstrak (misalnya PDF berbasis gambar).

        Raises:
            FileNotFoundError: Jika file PDF tidak ditemukan di path yang diberikan.
            fitz.FileDataError: Jika file PDF korup atau formatnya tidak valid.
            PermissionError:    Jika file PDF dilindungi password.
            RuntimeError:       Untuk error tak terduga lainnya saat membaca PDF.
        """
        pdf_path = Path(file_path)

        # --- Validasi keberadaan dan tipe file ---
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"File PDF tidak ditemukan: '{pdf_path.resolve()}'. "
                f"Pastikan file ada di direktori input yang benar."
            )

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Path yang diberikan bukan sebuah file: '{pdf_path}'"
            )

        # Beri peringatan jika ekstensi bukan .pdf, tapi tetap coba buka
        if pdf_path.suffix.lower() != ".pdf":
            logger.warning(
                f"Ekstensi file '{pdf_path.suffix}' bukan .pdf. "
                f"Tetap mencoba membuka sebagai dokumen PDF..."
            )

        logger.info(f"Membuka PDF: '{pdf_path.name}' ...")

        # --- Buka dokumen PDF ---
        try:
            doc: fitz.Document = fitz.open(str(pdf_path))
        except fitz.FileDataError as e:
            raise fitz.FileDataError(
                f"File PDF korup atau format tidak valid: '{pdf_path.name}'. "
                f"Detail: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Gagal membuka file PDF '{pdf_path.name}': {e}"
            ) from e

        # --- Cek proteksi password ---
        if doc.is_encrypted:
            doc.close()
            raise PermissionError(
                f"PDF '{pdf_path.name}' dilindungi password dan tidak dapat dibaca. "
                f"Harap hapus proteksi terlebih dahulu sebelum diproses."
            )

        total_pages: int = len(doc)
        logger.info(f"Total halaman terdeteksi: {total_pages}")

        # Kumpulkan teks tiap halaman dalam list lalu join di akhir
        # (lebih efisien dari konkatenasi string berulang pada loop besar)
        page_texts: list[str] = []

        for page_num in range(total_pages):
            try:
                page: fitz.Page = doc[page_num]
                # Mode "text" mengekstrak teks dalam urutan alami membaca (kiri ke kanan, atas ke bawah)
                page_text: str = page.get_text("text")  # type: ignore[attr-defined]

                if page_text.strip():
                    # Tambahkan penanda halaman untuk memudahkan debugging
                    page_texts.append(
                        f"\n--- [HALAMAN {page_num + 1}] ---\n{page_text}"
                    )
                else:
                    logger.warning(
                        f"Halaman {page_num + 1} kosong atau berbasis gambar "
                        f"(tidak ada teks yang dapat diekstrak via PyMuPDF). "
                        f"Pertimbangkan penggunaan OCR untuk halaman ini."
                    )
            except Exception as e:
                logger.error(
                    f"Gagal mengekstrak teks dari halaman {page_num + 1}: {e}. "
                    f"Halaman ini akan dilewati."
                )
                continue  # Lewati halaman bermasalah, lanjut ke halaman berikutnya

        doc.close()

        # Gabungkan semua teks halaman menjadi satu string
        raw_text: str = "\n".join(page_texts)

        if not raw_text.strip():
            logger.warning(
                f"Tidak ada teks yang berhasil diekstrak dari '{pdf_path.name}'. "
                f"PDF kemungkinan seluruhnya berbasis gambar (hasil scan). "
                f"Gunakan engine OCR (misalnya Tesseract) sebelum memproses dokumen ini."
            )
        else:
            logger.info(
                f"Ekstraksi teks berhasil. "
                f"Total: {len(raw_text):,} karakter dari {total_pages} halaman."
            )

        return raw_text

    # -------------------------------------------------------------------------
    # METHOD 2: clean_and_segment_text
    # -------------------------------------------------------------------------

    def clean_and_segment_text(self, raw_text: str) -> list[str]:
        """
        Membersihkan teks mentah dari residu dokumen lalu memisahkannya
        menjadi daftar butir soal individual yang bersih.

        ATURAN KRUSIAL PILIHAN GANDA:
            Teks opsi jawaban (A., B., C., D., E.) SELALU tetap menyatu
            dengan teks pertanyaan utamanya dalam satu elemen list.
            Opsi jawaban TIDAK akan terpotong menjadi elemen terpisah karena
            regex delimiter soal dirancang hanya mengenali awalan ANGKA (1–150),
            bukan huruf kapital (A–E).

        Tahapan Pembersihan (dilakukan secara berurutan):
            1. Hapus penanda batas halaman internal (--- [HALAMAN X] ---)
            2. Hapus pola nomor halaman dari dokumen asli
            3. Hapus karakter non-printable hasil artefak PDF/OCR
            4. Opsional: hapus baris yang hanya berisi angka (sisa penomoran hlm.)
            5. Normalisasi spasi berlebihan di dalam baris
            6. Normalisasi baris kosong berlebihan (maks. 2 baris kosong)
            7. Segmentasi: pecah teks menjadi butir soal per nomor
            8. Filter: buang segmen yang terlalu pendek atau tidak valid

        Args:
            raw_text: Teks mentah hasil dari `extract_raw_text()`.

        Returns:
            list[str]: Daftar string, setiap elemen adalah SATU butir soal
            yang bersih, termasuk opsi jawaban A–E jika ada.
            Mengembalikan list kosong [] jika tidak ada soal yang terdeteksi.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Input teks mentah kosong. Tidak ada yang dapat diproses.")
            return []

        logger.debug("Memulai proses pembersihan dan segmentasi teks...")
        text: str = raw_text

        # --- Langkah 1: Hapus penanda batas halaman internal ---
        # Menghapus marker yang kita sisipkan sendiri: "--- [HALAMAN N] ---"
        text = _RE_PAGE_MARKER.sub("", text)

        # --- Langkah 2: Hapus pola nomor halaman dari dokumen asli ---
        # Menangani: "Halaman 1 dari 5", "Page 2 of 10", "- 3 -", dst.
        text = _RE_PAGE_NUMBER.sub("", text)

        # --- Langkah 3: Hapus karakter non-printable / artefak encoding ---
        # Karakter aneh yang sering muncul dari PDF hasil scan atau copy-paste
        text = _RE_NON_PRINTABLE.sub("", text)

        # --- Langkah 4: Normalisasi spasi berlebihan dalam satu baris ---
        # Mengubah dua spasi atau lebih (dan tab) menjadi satu spasi
        text = _RE_EXCESS_SPACES.sub(" ", text)

        # --- Langkah 6: Normalisasi baris kosong berlebihan ---
        # Mengubah tiga baris kosong atau lebih menjadi dua baris kosong
        text = _RE_EXCESS_BLANK_LINES.sub("\n\n", text)
        text = text.strip()

        karakter_setelah_bersih: int = len(text)
        logger.debug(
            f"Teks setelah dibersihkan: "
            f"{karakter_setelah_bersih:,} karakter (dari {len(raw_text):,} sebelumnya)."
        )

        # --- Langkah 7: Segmentasi teks menjadi butir-butir soal ---
        # Menggunakan lookahead (?=...) agar pemisah (nomor soal) IKUT TERSIMPAN
        # di awal setiap segmen hasil split, tidak terhapus.
        #
        # Contoh hasil split untuk input "1. Soal A\nA. Opt\n2. Soal B":
        #   ["1. Soal A\nA. Opt", "2. Soal B"]
        #   ✓ Opsi "A. Opt" tetap menyatu dengan "1. Soal A"
        raw_segments: list[str] = _RE_QUESTION_DELIMITER.split(text)

        # --- Langkah 8: Pembersihan akhir dan filter segmen ---
        questions: list[str] = []
        diabaikan: int = 0

        for segment in raw_segments:
            cleaned_segment: str = segment.strip()

            # Filter 1: Buang segmen yang terlalu pendek (kurang dari min_question_length karakter)
            # Segmen pendek biasanya adalah sisa teks sebelum soal pertama (judul, header, dll.)
            if len(cleaned_segment) < self.min_question_length:
                logger.debug(
                    f"Segmen diabaikan (terlalu pendek, {len(cleaned_segment)} karakter): "
                    f"'{cleaned_segment[:40]}'"
                )
                diabaikan += 1
                continue

            # Filter 1.5: Buang segmen yang merupakan JUDUL SEKSI / HEADER DOKUMEN
            # Pola ini menangkap header seperti "PILIHAN GANDA", "URAIAN", "SOAL ESAI", dll.
            # yang lolos filter panjang tapi bukan soal sebenarnya.
            # Sebuah segmen dianggap header jika:
            # - Seluruh teksnya (pada baris pertama) hanya berisi huruf besar dan spasi
            # - DAN tidak diawali dengan pola nomor soal valid (angka diikuti . atau ) )
            first_line = cleaned_segment.split('\n')[0].strip()
            is_section_header = (
                re.match(r'^[A-Z\s]+$', first_line) and
                len(first_line) <= 60 and
                not _RE_VALID_QUESTION_START.match(cleaned_segment)
            )
            if is_section_header:
                # Header seksi biasanya diikuti soal berikutnya—gabungkan ke segmen soal pertama yang valid
                # Untuk sekarang, lewati saja header ini agar tidak terkirim ke LLM sebagai soal
                logger.warning(
                    f"Segmen diabaikan (judul seksi terdeteksi): '{first_line}'"
                )
                diabaikan += 1
                continue

            # Filter 2: HARD LIMIT CHUNKING (Pencegahan Resource Exhaustion)
            # Jika Regex gagal memecah dan menghasilkan 1 gumpalan raksasa,
            # sistem akan memaksakan pemotongan berdasarkan baris agar LLM tidak kehabisan memori.
            MAX_LENGTH = 1500
            if len(cleaned_segment) > MAX_LENGTH:
                logger.warning(
                    f"Segmen masif terdeteksi ({len(cleaned_segment)} karakter). "
                    f"Menerapkan pembatasan masukan ketat (Max {MAX_LENGTH} char)."
                )
                current_chunk = ""
                for line in cleaned_segment.split("\n"):
                    if len(current_chunk) + len(line) < MAX_LENGTH:
                        current_chunk += line + "\n"
                    else:
                        if len(current_chunk) >= self.min_question_length:
                            questions.append(current_chunk.strip())
                        
                        # Jika satu baris panjangnya abnormal, potong string secara harfiah
                        if len(line) >= MAX_LENGTH:
                            for i in range(0, len(line), MAX_LENGTH):
                                questions.append(line[i:i+MAX_LENGTH].strip())
                            current_chunk = ""
                        else:
                            current_chunk = line + "\n"
                            
                if len(current_chunk) >= self.min_question_length:
                    questions.append(current_chunk.strip())
            else:
                questions.append(cleaned_segment)

        logger.info(
            f"Segmentasi selesai. "
            f"Ditemukan {len(questions)} butir soal "
            f"({diabaikan} segmen pendek diabaikan)."
        )

        # Peringatan jika tidak ada soal yang terdeteksi sama sekali
        if not questions:
            logger.warning(
                "Tidak ada butir soal yang berhasil disegmentasi. "
                "Kemungkinan penyebab: format penomoran soal non-standar. "
                "Pola nomor soal yang didukung: "
                "'1. Teks', '1) Teks', 'No. 1 Teks', 'Soal 1. Teks'."
            )

        return questions

    # -------------------------------------------------------------------------
    # METHOD 3: process_document — Pipeline Utama (Sederhana)
    # -------------------------------------------------------------------------

    def process_document(self, file_path: str) -> list[str]:
        """
        Method utama: menjalankan pipeline pemrosesan PDF end-to-end.

        Memanggil `extract_raw_text()` lalu `clean_and_segment_text()` secara
        berurutan. Method ini dirancang untuk penggunaan sederhana di main.py
        dan modul lain yang hanya butuh list soal tanpa metadata tambahan.

        Args:
            file_path: Path (str) ke file PDF yang akan diproses.

        Returns:
            list[str]: Daftar butir soal yang sudah bersih dan tersegmentasi.
            Mengembalikan list kosong [] jika terjadi error atau tidak ada soal.
        """
        pdf_name: str = Path(file_path).name

        # Separator visual di log untuk memudahkan pembacaan log antar dokumen
        logger.info(f"{'='*60}")
        logger.info(f"Memulai pemrosesan dokumen: '{pdf_name}'")
        logger.info(f"{'='*60}")

        try:
            # Langkah 1: Ekstrak seluruh teks dari PDF
            raw_text: str = self.extract_raw_text(file_path)

            # Langkah 2: Bersihkan lalu segmentasi menjadi butir soal
            questions: list[str] = self.clean_and_segment_text(raw_text)

            logger.info(
                f"✓ Pemrosesan '{pdf_name}' selesai. "
                f"Hasil: {len(questions)} soal siap diklasifikasikan."
            )
            return questions

        except FileNotFoundError as e:
            logger.error(f"[FileNotFoundError] {e}")
            return []

        except PermissionError as e:
            logger.error(f"[PermissionError] {e}")
            return []

        except fitz.FileDataError as e:
            logger.error(f"[FileDataError] PDF korup atau tidak valid: {e}")
            return []

        except RuntimeError as e:
            logger.error(f"[RuntimeError] Kesalahan tak terduga saat membuka PDF: {e}")
            return []

        except Exception as e:
            logger.error(
                f"[UnexpectedError] Kesalahan tidak dikenal saat memproses "
                f"'{pdf_name}': {type(e).__name__}: {e}"
            )
            return []

    # -------------------------------------------------------------------------
    # METHOD 4: process_document_full — Pipeline Lengkap (Dengan Metadata)
    # -------------------------------------------------------------------------

    def process_document_full(self, file_path: str) -> ProcessedDocument:
        """
        Versi pipeline lengkap yang mengembalikan objek `ProcessedDocument`
        berisi metadata pemrosesan secara lengkap, bukan hanya list soal.

        Berguna ketika modul lain membutuhkan informasi tambahan seperti
        total halaman, nama file, teks mentah, atau status error secara terprogram.

        Args:
            file_path: Path (str) ke file PDF yang akan diproses.

        Returns:
            ProcessedDocument: Instance dengan semua atribut terisi.
                - Jika berhasil: `is_success == True`, `questions` berisi soal.
                - Jika gagal: `is_success == False`, `error_message` berisi deskripsi error.
        """
        pdf_path = Path(file_path)

        # Siapkan objek hasil dengan nilai default
        result = ProcessedDocument(
            file_path=str(pdf_path.resolve()),
            file_name=pdf_path.name,
        )

        try:
            # Buka sementara hanya untuk mendapatkan jumlah halaman
            with fitz.open(str(pdf_path)) as doc:
                result.total_pages = len(doc)

            # Jalankan pipeline ekstraksi dan segmentasi
            result.raw_text = self.extract_raw_text(file_path)
            result.questions = self.clean_and_segment_text(result.raw_text)

            logger.info(
                f"✓ process_document_full selesai: "
                f"'{result.file_name}' — {result.question_count} soal."
            )

        except FileNotFoundError as e:
            result.error_message = f"File tidak ditemukan: {e}"
            logger.error(result.error_message)

        except PermissionError as e:
            result.error_message = f"Akses ditolak (PDF terproteksi): {e}"
            logger.error(result.error_message)

        except fitz.FileDataError as e:
            result.error_message = f"PDF korup atau format tidak valid: {e}"
            logger.error(result.error_message)

        except Exception as e:
            result.error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Error tidak dikenal di process_document_full: {result.error_message}")

        return result

    # -------------------------------------------------------------------------
    # METHOD 5: batch_process — Pemrosesan Banyak File Sekaligus
    # -------------------------------------------------------------------------

    def batch_process(
        self,
        directory: Optional[str] = None,
        file_paths: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Memproses banyak file PDF sekaligus dari direktori atau daftar path eksplisit.

        Args:
            directory:  Path (str) ke direktori berisi file PDF.
                        Jika None, gunakan INPUT_DIR dari config.py.
            file_paths: List of str path ke file PDF yang spesifik.
                        Jika diberikan, argumen `directory` diabaikan.

        Returns:
            dict[str, list[str]]: Dictionary dengan nama file (str) sebagai kunci
            dan list butir soal (list[str]) sebagai nilai.
            Contoh: {"soal_matematika.pdf": ["1. Tentukan nilai x...", ...], ...}
        """
        # Tentukan daftar file yang akan diproses
        if file_paths is not None:
            pdf_files: list[Path] = [Path(p) for p in file_paths]
        else:
            target_dir = Path(directory) if directory else INPUT_DIR
            pdf_files = sorted(target_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(
                "Tidak ada file PDF yang ditemukan untuk diproses secara batch."
            )
            return {}

        logger.info(f"Batch processing dimulai: {len(pdf_files)} file PDF ditemukan.")
        results: dict[str, list[str]] = {}

        for i, pdf_file in enumerate(pdf_files, start=1):
            logger.info(f"[{i}/{len(pdf_files)}] Memproses: '{pdf_file.name}'")
            questions = self.process_document(str(pdf_file))
            results[pdf_file.name] = questions

        total_questions_extracted = sum(len(q) for q in results.values())
        logger.info(
            f"Batch processing selesai. "
            f"{len(results)} dokumen berhasil diproses, "
            f"total {total_questions_extracted} soal diekstrak."
        )
        return results


# =============================================================================
# PENGUJIAN MANDIRI (jalankan: python modules/pdf_processor.py)
# =============================================================================
if __name__ == "__main__":
    import sys

    logger.info("=== Test Mandiri: modules/pdf_processor.py ===")

    # Inisialisasi processor dengan konfigurasi default
    processor = PDFProcessor(min_question_length=15, strip_lone_numbers=True)

    # Cari semua PDF di folder input
    input_folder = Path("data/input")
    pdf_list = list(input_folder.glob("*.pdf"))

    if not pdf_list:
        logger.warning(
            f"Tidak ada file .pdf di '{input_folder.resolve()}'. "
            f"Letakkan file PDF soal ujian di folder tersebut untuk menguji modul ini."
        )
        sys.exit(0)

    # Proses setiap PDF yang ditemukan
    for pdf_file in pdf_list:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Menguji: {pdf_file.name}")

        # Gunakan versi lengkap untuk mendapatkan metadata
        doc_result: ProcessedDocument = processor.process_document_full(str(pdf_file))

        if doc_result.is_success:
            logger.info(f"  Status    : Berhasil ✓")
            logger.info(f"  Halaman   : {doc_result.total_pages}")
            logger.info(f"  Jumlah soal: {doc_result.question_count}")

            # Tampilkan 3 soal pertama sebagai preview hasil segmentasi
            logger.info("  Preview 3 soal pertama:")
            for idx, q in enumerate(doc_result.questions[:3], start=1):
                # Tampilkan max 120 karakter dan ganti newline dengan spasi untuk log
                preview = q[:120].replace("\n", " ↵ ")
                logger.info(f"    [{idx}] {preview}...")
        else:
            logger.error(f"  Status  : Gagal ✗")
            logger.error(f"  Error   : {doc_result.error_message}")
