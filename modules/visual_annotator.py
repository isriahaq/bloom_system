# =============================================================================
# modules/visual_annotator.py
# Anotasi Visual PDF berdasarkan Klasifikasi Taksonomi Bloom
# =============================================================================
#
# Modul ini bertugas sebagai lapisan presentasi akhir dari pipeline.
# Ia menerima hasil evaluasi dari HybridLLMRouter dan menerapkan highlight
# berwarna pada teks soal di dokumen PDF asli menggunakan PyMuPDF (fitz).
#
# Peta Warna Highlight:
#   - LOTS (C1)    : Hijau  — mudah dijawab, tidak menantang
#   - MOTS (C2, C3): Kuning — tingkat menengah
#   - HOTS (C4-C6) : Merah/Oranye — menuntut kemampuan berpikir tingkat tinggi
#
# Digunakan oleh: main.py / pipeline utama
# Bergantung pada: PyMuPDF (pip install PyMuPDF)
# =============================================================================
# CHANGELOG (Revisi 3):
#   - Pisahkan _extract_stem_lines() sebagai method tersendiri
#   - Tambah _search_phrase_in_page() dengan 5 strategi toleransi
#   - Tambah ultra-fallback: pencarian substring kata (4-kata)
#   - Auto-repair kategori dari level langsung di dalam loop anotasi
#   - Guard clause diperlonggar: soal dengan level valid tidak di-block
#   - Soal ERROR di-skip hanya jika BENAR-BENAR tidak ada level/kategori apapun

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any

# --- Impor PyMuPDF ---
# PyMuPDF di-import sebagai 'fitz'. Pastikan sudah terinstall:
#   pip install PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None  # type: ignore[assignment]

# --- Impor logger dari utilitas proyek ---
try:
    from utils.logger import logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger(__name__)  # type: ignore[assignment]


# =============================================================================
# BAGIAN 1: KONSTANTA WARNA DAN KONFIGURASI
# =============================================================================

# Peta warna highlight dalam format RGB (0.0 – 1.0) untuk setiap kategori Bloom.
# Format: (Red, Green, Blue) — semua nilai antara 0.0 dan 1.0
HIGHLIGHT_COLORS: dict[str, tuple[float, float, float]] = {
    # Hijau terang untuk LOTS (Lower Order Thinking Skills) — C1, C2
    "LOTS": (0.56, 0.93, 0.56),   # Light Green
    # Kuning terang untuk MOTS (Middle Order Thinking Skills) — C3
    "MOTS": (1.00, 0.97, 0.40),   # Light Yellow
    # Oranye kemerahan untuk HOTS (Higher Order Thinking Skills) — C4, C5, C6
    "HOTS": (1.00, 0.55, 0.20),   # Orange-Red
    # Warna default jika kategori tidak dikenali (abu-abu)
    "DEFAULT": (0.80, 0.80, 0.80),
}

# Opacity (alpha) highlight. Nilai 1.0 = penuh, 0.0 = transparan total.
# Nilai ~0.4 memberi efek highlight yang terlihat namun teks tetap terbaca.
HIGHLIGHT_OPACITY: float = 0.4

# Panjang minimum anchor teks untuk pencarian parsial
MIN_ANCHOR_LEN: int = 8

# Pola regex untuk mendeteksi awal baris opsi jawaban pilihan ganda
_OPTION_PAT = re.compile(
    r"^(?:"
    r"[A-Ea-e][\.)][\t ]"
    r"|[A-Ea-e]\s*[-\u2013]\s+"
    r"|\([A-Ea-e]\)\s+"
    r")",
    re.UNICODE,
)

# Pola regex untuk mendeteksi baris kunci jawaban
# Menangani berbagai format: "Kunci:", "Kunci Jawaban:", "Kunci Jawaban : B", "Jawab:", dll.
_KEY_PAT = re.compile(
    r"^(?:"
    r"kunci\s*jawaban?"       # Kunci Jawaban / Kunci Jawab
    r"|kunci"                 # Kunci:
    r"|jawab(?:an)?"          # Jawab / Jawaban
    r"|answer"
    r"|pembahasan"
    r"|kj"                    # KJ (singkatan)
    r")\s*[:\uff1a]",
    re.IGNORECASE | re.UNICODE,
)

# Pola untuk mendeteksi apakah baris adalah akhir-kalimat generic yang pendek.
# Baris seperti "adalah:", "ini:", "berikut:" yang hanya berisi kata penutup kalimat
# TIDAK dicari secara mandiri karena terlalu umum dan menyebabkan false-positive.
_TRAILING_FRAGMENT_PAT = re.compile(
    r"^(?:[\w\s]{0,30}\s+)?(?:adalah|ialah|yaitu|berikut|merupakan|tersebut)[:\s]*$",
    re.IGNORECASE | re.UNICODE,
)



# =============================================================================
# BAGIAN 2: DATACLASS UNTUK HASIL ANOTASI
# =============================================================================

@dataclass
class AnnotationResult:
    """
    Menyimpan ringkasan hasil proses anotasi satu dokumen PDF.

    Attributes:
        output_path:        Path file PDF hasil anotasi.
        total_questions:    Jumlah total soal yang hendak dianotasi.
        annotated_count:    Jumlah soal yang berhasil ditemukan dan dianotasi.
        not_found_count:    Jumlah soal yang teksnya tidak ditemukan di PDF.
        error_count:        Jumlah soal yang menemui error saat pemrosesan.
        not_found_texts:    Daftar teks soal yang gagal ditemukan (untuk debugging).
    """
    output_path: str = ""
    total_questions: int = 0
    annotated_count: int = 0
    not_found_count: int = 0
    error_count: int = 0
    ignored_count: int = 0
    not_found_texts: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Persentase soal yang berhasil dianotasi."""
        if self.total_questions == 0:
            return 0.0
        return (self.annotated_count / self.total_questions) * 100


# =============================================================================
# BAGIAN 3: CLASS UTAMA PDFAnnotator
# =============================================================================

class PDFAnnotator:
    """
    Menerapkan anotasi highlight berwarna pada dokumen PDF berdasarkan
    hasil klasifikasi Taksonomi Bloom dari HybridLLMRouter.

    Kelas ini berperan sebagai lapisan akhir dari pipeline "Turnitin for HOTS":
    soal-soal dalam PDF asli diwarnai sesuai levelnya sehingga guru atau
    validator dapat langsung melihat distribusi LOTS/MOTS/HOTS secara visual.

    Penggunaan dasar:
        annotator = PDFAnnotator()
        result = annotator.annotate_document(
            input_pdf_path="data/soal.pdf",
            output_pdf_path="data/output/soal_anotasi.pdf",
            evaluation_results=[
                {
                    "question_text": "Sebutkan organ pernapasan manusia!",
                    "KKO": "sebutkan",
                    "Level": "C1",
                    "Kategori": "LOTS",
                    "model_used": "gemini-1.5-flash",
                },
                ...
            ]
        )
        print(f"Anotasi selesai: {result.annotated_count}/{result.total_questions} soal")
    """

    def __init__(self) -> None:
        """Inisialisasi PDFAnnotator dan validasi ketersediaan library PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "Library PyMuPDF tidak ditemukan. "
                "Silakan install dengan perintah: pip install PyMuPDF"
            )
        logger.info("PDFAnnotator berhasil diinisialisasi.")

    # -------------------------------------------------------------------------
    # METHOD PRIVAT: UTILITAS INTERNAL
    # -------------------------------------------------------------------------

    def _get_highlight_color(self, kategori: str) -> tuple[float, float, float]:
        """
        Mengembalikan warna highlight (RGB) berdasarkan kategori Bloom.

        Args:
            kategori: String kategori — "LOTS", "MOTS", "HOTS", atau lainnya.

        Returns:
            Tuple (R, G, B) dengan nilai antara 0.0 dan 1.0.
        """
        kategori_upper = str(kategori).upper().strip()
        return HIGHLIGHT_COLORS.get(kategori_upper, HIGHLIGHT_COLORS["DEFAULT"])

    def _build_popup_comment(self, eval_item: dict[str, Any]) -> str:
        """
        Membuat teks komentar/tooltip untuk popup anotasi pada highlight.

        Komentar ini akan muncul saat pengguna mengarahkan kursor ke highlight
        di PDF reader. Format yang dihasilkan:
            [HOTS - C6] KKO: Merancang | Dinilai oleh: gemini-1.5-flash

        Args:
            eval_item: Dictionary hasil evaluasi dari HybridLLMRouter,
                       berisi kunci: Kategori, Level, KKO, model_used.

        Returns:
            String teks komentar yang siap disisipkan ke anotasi.
        """
        kategori  = eval_item.get("kategori", eval_item.get("Kategori", "N/A"))
        level     = eval_item.get("level", eval_item.get("Level", "N/A"))
        kko       = eval_item.get("kko", eval_item.get("KKO", "N/A"))
        model     = eval_item.get("model_used",  "unknown")

        return f"[{kategori} - {level}]\nKKO: {kko}\nDinilai oleh: {model}"

    def _normalize_text(self, text: str) -> str:
        """
        Menormalkan teks dengan menyederhanakan spasi/baris baru yang berlebihan.

        PDF sering menyimpan teks dengan spasi atau newline yang tidak konsisten
        dibanding teks yang dihasilkan oleh LLM/OCR. Normalisasi diperlukan
        untuk meningkatkan keberhasilan pencarian teks.

        Args:
            text: Teks mentah yang akan dinormalisasi.

        Returns:
            Teks yang sudah dibersihkan spasi dan newline-nya.
        """
        # Ganti semua whitespace (newline, tab, spasi ganda) dengan satu spasi
        normalized = re.sub(r"\s+", " ", text)
        return normalized.strip()

    def _sanitize_quotes(self, text: str) -> str:
        """Membersihkan kutip ganda yang di-escape CSV (""teks"" → "teks")."""
        return text.replace('""', '"')

    def _extract_stem_lines(self, question_text: str) -> list[str]:
        """
        Mengekstrak baris-baris BATANG SOAL (stem) saja dari teks soal lengkap.

        Berhenti saat menemukan baris opsi jawaban (A., B., C.) atau kunci jawaban.
        Mengembalikan SEMUA baris stem tanpa merging — filtering panjang dilakukan
        di _search_text_in_page agar baris panjang dan pendek diperlakukan berbeda.

        Returns:
            list[str]: Semua baris yang merupakan bagian dari stem soal.
        """
        raw_stem: list[str] = []
        for raw_line in question_text.split('\n'):
            line = raw_line.strip()
            if not line:
                continue
            if _OPTION_PAT.match(line) or _KEY_PAT.match(line):
                break
            raw_stem.append(line)

        # Safety: jika tidak ada stem (soal esai tanpa nomor), ambil semua non-opsi
        if not raw_stem:
            for raw_line in question_text.split('\n'):
                line = raw_line.strip()
                if line and not _OPTION_PAT.match(line) and not _KEY_PAT.match(line):
                    raw_stem.append(line)

        return raw_stem


    def _search_phrase_in_page(self, page: Any, phrase: str) -> list[Any]:
        """
        Mencari satu frasa di halaman PDF dengan 5 strategi toleransi.

        Strategi (berurutan, berhenti saat berhasil):
          1. Exact match
          2. Normalized (spasi/unicode diseragamkan)
          3. Anchor 60 karakter pertama
          4. Anchor 40 karakter pertama
          5. Anchor 25 karakter pertama (fallback paling toleran)

        Returns:
            list[fitz.Quad]: Lokasi teks ditemukan, atau [] jika tidak ada.
        """
        # Strategi 1: Exact
        found = page.search_for(phrase, quads=True)
        if found:
            return found

        # Strategi 2: Normalized
        norm = self._normalize_text(phrase)
        if norm and norm != phrase:
            found = page.search_for(norm, quads=True)
            if found:
                return found

        # Strategi 3-5: Anchor dengan panjang menurun
        for anchor_len in (60, 40, 25):
            if len(phrase) > anchor_len:
                anchor = self._normalize_text(phrase[:anchor_len])
                if len(anchor) >= MIN_ANCHOR_LEN:
                    found = page.search_for(anchor, quads=True)
                    if found:
                        return found

        return []

    def _search_text_in_page(
        self,
        page: Any,  # fitz.Page
        question_text: str,
        kko: str = "",
    ) -> list[Any]:  # list[fitz.Quad]
        """
        Mencari lokasi BATANG SOAL di dalam halaman PDF untuk di-highlight.

        Aturan:
          - Hanya sorot batang soal (kalimat pertanyaan + nomor soal).
          - TIDAK menyorot opsi jawaban (A./B./C./D./E.) maupun
            baris kunci jawaban (Jawab:, Kunci:, dll.).

        Strategi (berurutan):
          1. Cari setiap stem line yang cukup panjang (>= 15 char) dengan
             _search_phrase_in_page (5 strategi toleransi).
          2. Fallback kombinasi: gabungkan semua stem line jadi satu teks
             dan cari anchor 60/40/25 karakter di tengah kalimat.
          3. Ultra-fallback: word-snippet 4-kata dari semua posisi stem
             (bukan hanya 6 posisi awal) untuk mengatasi kasus di mana
             awal kalimat mengandung nomor soal yang tersimpan berbeda di PDF.
        """
        question_text = self._sanitize_quotes(question_text)

        # Langkah 1: Ekstrak stem lines (tanpa opsi, tanpa kunci jawaban)
        stem_lines = self._extract_stem_lines(question_text)
        if not stem_lines:
            return []

        # Langkah 2: Cari setiap stem line yang CUKUP PANJANG
        # Minimum 15 karakter: mencegah fragmen pendek seperti "adalah:"
        # yang merupakan akhiran kalimat ter-wrap di-highlight tersendiri.
        all_instances: list[Any] = []
        for line in stem_lines:
            if len(line) < 15:
                continue
            # Skip baris yang merupakan trailing generic fragment
            if _TRAILING_FRAGMENT_PAT.match(line):
                continue
            instances = self._search_phrase_in_page(page, line)
            if instances:
                all_instances.extend(instances)

        # Langkah 3: Fallback — cari frasa gabungan dari stem lines substantif
        # Berguna ketika setiap baris PDF terlalu pendek untuk ditemukan tersendiri.
        if not all_instances and stem_lines:
            # Gunakan hanya baris yang cukup panjang untuk combined search
            long_lines = [l for l in stem_lines if len(l) >= 10]
            if long_lines:
                # Coba gabungan 3 baris pertama
                combined = " ".join(long_lines[:3])
                if len(combined) >= 20:
                    instances = self._search_phrase_in_page(page, combined)
                    if instances:
                        all_instances.extend(instances)

        # Langkah 4: Ultra-fallback — word-snippet 4-kata dari semua posisi
        # Ini sangat efektif untuk soal yang nomor awalnya tersimpan terpisah di PDF.
        # Kita mulai dari posisi 1 (lewati nomor soal di awal) untuk hit rate lebih baik.
        if not all_instances and stem_lines:
            for line in stem_lines:
                if len(line) < 10:
                    continue
                words = line.split()
                if len(words) < 4:
                    continue
                # Coba dari posisi 1 dahulu (skip nomor soal), lalu 0, 2, dst.
                positions = list(range(1, min(len(words) - 3, 10))) + [0]
                for start_w in positions:
                    if start_w >= len(words) - 3:
                        continue
                    snippet = " ".join(words[start_w:start_w + 4])
                    if len(snippet) >= 10:
                        found = page.search_for(snippet, quads=True)
                        if found:
                            all_instances.extend(found)
                            break
                if all_instances:
                    break

        return all_instances

    def _apply_highlight(

        self,
        page: Any,   # fitz.Page
        instances: list[Any],  # list[fitz.Rect]
        color: tuple[float, float, float],
        comment: str,
    ) -> int:
        """
        Menerapkan anotasi highlight pada semua instance (lokasi) teks yang ditemukan.

        Setiap instance (baris teks) diberi highlight terpisah dengan warna dan
        komentar tooltip yang sama.

        Args:
            page:      Objek halaman PDF (fitz.Page).
            instances: List kotak (fitz.Rect) dari lokasi teks hasil search_for.
            color:     Tuple (R, G, B) warna highlight.
            comment:   Teks tooltip/popup yang disisipkan ke anotasi.

        Returns:
            Jumlah highlight yang berhasil diterapkan (biasanya = len(instances)).
        """
        applied_count = 0
        for rect in instances:
            try:
                # Buat anotasi highlight pada kotak yang ditemukan
                highlight = page.add_highlight_annot(rect)

                # Atur warna highlight menggunakan nilai RGB
                highlight.set_colors(stroke=color)

                # Atur opacity agar teks di bawahnya tetap terlihat
                highlight.set_opacity(HIGHLIGHT_OPACITY)

                # Sisipkan komentar sebagai popup info (muncul saat diklik/hover)
                highlight.set_info(content=comment)

                # Perbarui anotasi agar perubahan tersimpan ke objek halaman
                highlight.update()
                applied_count += 1

            except Exception as e:
                logger.warning(f"  Gagal menerapkan highlight pada satu rect: {e}")

        return applied_count

    # -------------------------------------------------------------------------
    # METHOD PUBLIK: ENTRY POINT UTAMA
    # -------------------------------------------------------------------------

    def annotate_document(
        self,
        input_pdf_path: str,
        output_pdf_path: str,
        evaluation_results: list[dict[str, Any]],
    ) -> AnnotationResult:
        """
        Anotasi utama: menerapkan highlight berwarna pada PDF berdasarkan hasil evaluasi.

        Ini adalah method entry point yang dipanggil oleh pipeline utama (main.py).
        Method ini:
        1. Membuka PDF asli.
        2. Untuk setiap soal dalam evaluation_results, mencari teksnya di PDF.
        3. Menerapkan highlight berwarna sesuai level Bloom-nya.
        4. Menyisipkan tooltip/komentar pada setiap highlight.
        5. Menyimpan PDF yang sudah dianotasi ke output_pdf_path.

        Args:
            input_pdf_path:     Path absolut/relatif ke file PDF asli (input).
            output_pdf_path:    Path absolut/relatif untuk menyimpan file PDF hasil
                                anotasi. Direktori akan dibuat otomatis jika belum ada.
            evaluation_results: List of dictionaries, satu dict per soal, berisi:
                                 - "question_text" (str): Teks soal asli.
                                 - "KKO"           (str): Kata Kerja Operasional.
                                 - "Level"         (str): Level Bloom (C1–C6).
                                 - "Kategori"      (str): "LOTS", "MOTS", atau "HOTS".
                                 - "model_used"    (str): Nama model LLM yang menilai.
                                 (Kunci lain akan diabaikan.)

        Returns:
            AnnotationResult: Objek ringkasan berisi statistik proses anotasi.

        Raises:
            FileNotFoundError: Jika input_pdf_path tidak ditemukan.
            ValueError:        Jika evaluation_results kosong atau tidak valid.
            RuntimeError:      Jika file PDF tidak dapat dibuka oleh PyMuPDF.

        Example:
            >>> annotator = PDFAnnotator()
            >>> result = annotator.annotate_document(
            ...     input_pdf_path="data/soal_ujian.pdf",
            ...     output_pdf_path="data/output/soal_anotasi.pdf",
            ...     evaluation_results=[
            ...         {
            ...             "question_text": "Hitunglah luas persegi panjang ...",
            ...             "KKO": "hitunglah",
            ...             "Level": "C3",
            ...             "Kategori": "MOTS",
            ...             "model_used": "gemini-1.5-flash",
            ...         }
            ...     ]
            ... )
            >>> print(result.success_rate)  # mis. 95.0
        """
        # ------------------------------------------------------------------ #
        # VALIDASI INPUT                                                       #
        # ------------------------------------------------------------------ #
        if not os.path.isfile(input_pdf_path):
            raise FileNotFoundError(
                f"File PDF input tidak ditemukan: '{input_pdf_path}'"
            )

        if not evaluation_results:
            raise ValueError(
                "evaluation_results tidak boleh kosong. "
                "Pastikan pipeline LLM sudah berjalan sebelum memanggil annotate_document."
            )

        # ------------------------------------------------------------------ #
        # GUARD CLAUSE: PASTIKAN ADA MINIMAL SATU SOAL VALID                 #
        # (diperlonggar: soal dengan level valid tapi kategori ERROR masih OK)#
        # ------------------------------------------------------------------ #
        _VALID_CATS_GC  = {"LOTS", "MOTS", "HOTS"}
        _VALID_LEVELS_GC = {"C1", "C2", "C3", "C4", "C5", "C6"}

        def _is_annotatable(item: dict) -> bool:
            qt  = str(item.get("question_text", "")).strip()
            if not qt:
                return False
            kat = str(item.get("kategori", item.get("Kategori", ""))).strip().upper()
            lvl = str(item.get("level",    item.get("Level",    ""))).strip().upper()
            if kat in _VALID_CATS_GC:
                return True
            if lvl in _VALID_LEVELS_GC:
                return True
            return False

        has_valid_category = any(_is_annotatable(item) for item in evaluation_results)

        if not has_valid_category:
            logger.error("PDFAnnotator: Guard Clause aktif - Tidak ada data kognitif valid dalam hasil evaluasi, membatalkan anotasi.")
            return AnnotationResult(
                output_path=output_pdf_path,
                total_questions=len(evaluation_results),
                annotated_count=0,
                not_found_count=0,
                error_count=len(evaluation_results)
            )

        # Inisialisasi objek hasil anotasi
        result = AnnotationResult(
            output_path=output_pdf_path,
            total_questions=len(evaluation_results),
        )

        # Pastikan direktori output ada; buat jika belum ada
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"PDFAnnotator: Memulai proses anotasi dokumen")
        logger.info(f"  Input PDF  : {input_pdf_path}")
        logger.info(f"  Output PDF : {output_pdf_path}")
        logger.info(f"  Jumlah soal: {result.total_questions}")
        logger.info("=" * 60)

        # ------------------------------------------------------------------ #
        # BUKA DOKUMEN PDF MENGGUNAKAN PYMUPDF                                #
        # ------------------------------------------------------------------ #
        try:
            doc = fitz.open(input_pdf_path)
        except Exception as e:
            raise RuntimeError(
                f"PyMuPDF gagal membuka file PDF '{input_pdf_path}': {e}"
            ) from e

        logger.info(f"  Dokumen terbuka: {doc.page_count} halaman.")

        # ------------------------------------------------------------------ #
        # PROSES SETIAP SOAL: CARI TEKS & TERAPKAN HIGHLIGHT                 #
        # ------------------------------------------------------------------ #
        for idx, eval_item in enumerate(evaluation_results, start=1):
            if not eval_item.get("is_question", True):
                logger.info(f"  [Soal #{idx}] Ditandai sebagai bukan soal, diabaikan dari anotasi.")
                result.ignored_count += 1
                continue
                
            question_text = eval_item.get("question_text", "").strip()
            kategori      = str(eval_item.get("kategori", eval_item.get("Kategori", ""))).strip().upper()
            level         = str(eval_item.get("level",    eval_item.get("Level",    ""))).strip().upper()
            kko           = str(eval_item.get("kko", eval_item.get("KKO", ""))).strip()
            model_used    = str(eval_item.get("model_used", "")).lower()

            # Lewati soal yang tidak memiliki teks soal
            if not question_text:
                logger.warning(f"  [Soal #{idx}] Teks soal kosong, dilewati.")
                result.error_count += 1
                continue

            # --- AUTO-REPAIR: Infer kategori dari level jika kategori tidak valid ---
            _VALID_CATS_LOOP = {"LOTS", "MOTS", "HOTS"}
            _VALID_LEVELS_LOOP = {"C1", "C2", "C3", "C4", "C5", "C6"}
            if kategori not in _VALID_CATS_LOOP:
                # Coba infer dari level
                if level == "C1":
                    kategori = "LOTS"
                elif level in ("C2", "C3"):
                    kategori = "MOTS"
                elif level in ("C4", "C5", "C6"):
                    kategori = "HOTS"
                else:
                    logger.warning(
                        f"  [Soal #{idx}] Kategori tidak valid ('{kategori}') dan level "
                        f"tidak dikenali ('{level}'). Dilewati. "
                        f"Teks: '{question_text[:60]}...'"
                    )
                    result.error_count += 1
                    continue
                logger.info(
                    f"  [Soal #{idx}] Kategori di-infer dari level '{level}' → '{kategori}'"
                )
                eval_item["kategori"] = kategori

            # Tentukan warna dan buat teks komentar popup
            color   = self._get_highlight_color(kategori)
            comment = self._build_popup_comment(eval_item)

            logger.info(
                f"  [Soal #{idx}/{result.total_questions}] "
                f"[{kategori} - {level}] "
                f"'{question_text[:50]}...'"
            )

            # --- Cari teks soal di semua halaman PDF ---
            found_on_any_page = False

            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    instances = self._search_text_in_page(page, question_text, kko)

                    if instances:
                        # Terapkan highlight pada semua baris yang cocok
                        applied = self._apply_highlight(page, instances, color, comment)
                        logger.info(
                            f"    ✓ Ditemukan di halaman {page_num + 1}: "
                            f"{len(instances)} lokasi, {applied} highlight diterapkan."
                        )
                        found_on_any_page = True
                except Exception as e:
                    logger.warning(f"    ⚠ Gagal memproses anotasi halaman {page_num + 1} soal #{idx}: {e}")
                    continue

            if not found_on_any_page:
                # Teks tidak ditemukan di seluruh dokumen — catat untuk laporan
                short_text = question_text[:80]
                logger.warning(
                    f"    ✗ Teks soal #{idx} TIDAK DITEMUKAN di PDF. "
                    f"Cuplikan: '{short_text}...'"
                )
                result.not_found_count += 1
                result.not_found_texts.append(question_text[:120])
            else:
                result.annotated_count += 1

        # ------------------------------------------------------------------ #
        # SIMPAN DOKUMEN YANG SUDAH DIANOTASI                                 #
        # ------------------------------------------------------------------ #
        try:
            doc.save(output_pdf_path, garbage=4, deflate=True)
            logger.info(f"  PDF hasil anotasi disimpan ke: '{output_pdf_path}'")
        except Exception as e:
            doc.close()
            raise RuntimeError(
                f"Gagal menyimpan PDF ke '{output_pdf_path}': {e}"
            ) from e
        finally:
            doc.close()

        # ------------------------------------------------------------------ #
        # LAPORAN AKHIR                                                        #
        # ------------------------------------------------------------------ #
        logger.info("=" * 60)
        logger.info(f"PDFAnnotator: Proses anotasi selesai.")
        logger.info(f"  Berhasil dianotasi : {result.annotated_count} soal")
        logger.info(f"  Tidak ditemukan    : {result.not_found_count} soal")
        logger.info(f"  Error/dilewati     : {result.error_count} soal")
        logger.info(f"  Diabaikan (bukan)  : {result.ignored_count} soal")
        logger.info(f"  Tingkat keberhasilan: {result.success_rate:.1f}%")
        logger.info("=" * 60)

        # Log soal yang tidak berhasil ditemukan (untuk debugging)
        if result.not_found_texts:
            logger.warning(
                "  Daftar soal yang tidak ditemukan di PDF "
                "(kemungkinan perbedaan format/encoding):"
            )
            for i, text in enumerate(result.not_found_texts, 1):
                logger.warning(f"    {i}. '{text}'")

        return result


# =============================================================================
# BAGIAN 4: FUNGSI UTILITAS TINGKAT MODUL
# =============================================================================

def get_color_legend() -> dict[str, dict[str, Any]]:
    """
    Mengembalikan legenda warna yang digunakan oleh annotator.

    Berguna untuk ditampilkan di UI atau di laporan PDF akhir.

    Returns:
        Dictionary berformat:
        {
            "LOTS": {"color_rgb": (R, G, B), "levels": ["C1", "C2"], "label": "..."},
            ...
        }
    """
    return {
        "LOTS": {
            "color_rgb": HIGHLIGHT_COLORS["LOTS"],
            "levels": ["C1"],
            "label": "Lower Order Thinking Skills (Mengingat)",
            "hex_approx": "#8EED8E",
        },
        "MOTS": {
            "color_rgb": HIGHLIGHT_COLORS["MOTS"],
            "levels": ["C2", "C3"],
            "label": "Middle Order Thinking Skills (Memahami, Mengaplikasikan)",
            "hex_approx": "#FFF766",
        },
        "HOTS": {
            "color_rgb": HIGHLIGHT_COLORS["HOTS"],
            "levels": ["C4", "C5", "C6"],
            "label": "Higher Order Thinking Skills (Menganalisis, Mengevaluasi, Mencipta)",
            "hex_approx": "#FF8C33",
        },
    }
