# =============================================================================
# app.py
# Sistem Klasifikasi Taksonomi Bloom (HOTS)
# Antarmuka Web Utama — Streamlit
# =============================================================================
# Cara menjalankan:
#   streamlit run app.py
# =============================================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import importlib
import modules.pdf_processor
import modules.llm_router
import modules.visual_annotator

importlib.reload(modules.pdf_processor)
importlib.reload(modules.llm_router)
importlib.reload(modules.visual_annotator)

from modules.pdf_processor import PDFProcessor
from modules.llm_router import CloudLLMRouter, HybridLLMRouter
from modules.visual_annotator import PDFAnnotator

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================

st.set_page_config(
    page_title="Sistem Klasifikasi Taksonomi Bloom",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS minimal — hanya untuk memperlebar baris tabel dan warna header
st.markdown("""
<style>
/* Tinggi baris metrik */
[data-testid="stMetric"] { padding: 0.6rem 0.8rem; }
/* Kapital judul kartu */
.card-title { font-weight: 600; margin-bottom: 0.4rem; font-size: 1.02rem; }
/* Lebar default dataframe lebih lega */
[data-testid="stDataFrame"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("Panduan")
    st.markdown("""
    1. Unggah file PDF soal ujian
    2. Klik **Mulai Analisis Dokumen**
    3. Tunggu proses evaluasi selesai
    4. Lihat hasil dan unduh PDF teranotasi

    ---
    **Keterangan Warna di PDF:**

    Hijau — LOTS\n
    Kuning — MOTS\n
    Oranye — HOTS

    ---
    | Level | Kategori |
    |---|---|
    | C1 Mengingat | LOTS |
    | C2 Memahami | MOTS |
    | C3 Mengaplikasikan | MOTS |
    | C4 Menganalisis | HOTS |
    | C5 Mengevaluasi | HOTS |
    | C6 Mencipta | HOTS |
    """, unsafe_allow_html=True)


# =============================================================================
# FUNGSI UTILITAS
# =============================================================================

def simpan_file_upload(uploaded_file) -> str:
    """Menyimpan buffer upload ke disk, verifikasi ukuran, kembalikan path."""
    input_dir = ROOT_DIR / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    save_path = input_dir / "temp_upload.pdf"
    with open(str(save_path), "wb") as f:
        f.write(uploaded_file.getbuffer())
    if not save_path.is_file() or save_path.stat().st_size == 0:
        raise IOError(f"File gagal disimpan ke '{save_path}'.")
    return str(save_path)


def hitung_statistik(df: pd.DataFrame) -> dict:
    """Hitung distribusi Bloom dari DataFrame hasil evaluasi."""
    # Hanya hitung yang bukan "N/A" di Kategori (artinya hanya soal valid)
    df_valid = df[df["Kategori"].str.upper() != "N/A"]
    total = len(df_valid)
    
    if total == 0:
        empty = {"count": 0, "pct": 0.0}
        return {
            "total": 0,
            "LOTS": empty, "MOTS": empty,
            "HOTS": empty, "ERROR": empty,
            "gemini_count": 0, "ollama_count": 0,
        }
    stats: dict = {"total": total}
    for kat in ["LOTS", "MOTS", "HOTS", "ERROR"]:
        n = int((df_valid["Kategori"].str.upper() == kat).sum())
        stats[kat] = {"count": n, "pct": round(n / total * 100, 1)}
    stats["gemini_count"] = int(
        df["Model"].str.contains("gemini|flash|pro", case=False, na=False).sum()
    )
    return stats


# =============================================================================
# HEADER
# =============================================================================

st.title("Sistem Klasifikasi Taksonomi Bloom")
st.caption(
    "Analisis distribusi tingkat kognitif instrumen asesmen secara otomatis"
)
st.divider()


# =============================================================================
# KARTU: UNGGAH DOKUMEN
# =============================================================================

with st.container(border=True):
    st.markdown('<p class="card-title">Unggah Dokumen Soal</p>', unsafe_allow_html=True)

    col_up, col_hint = st.columns([3, 2], gap="large")

    with col_up:
        uploaded_file = st.file_uploader(
            label="Pilih file PDF (soal ujian)",
            type=["pdf"],
            help="PDF dengan teks yang dapat disalin. Bukan hasil scan.",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.success(
                f"**{uploaded_file.name}** siap dianalisis "
                f"({uploaded_file.size / 1024:.1f} KB)"
            )
        else:
            st.info("Seret file PDF ke sini atau klik untuk memilih file.")

    with col_hint:
        st.markdown("""
        **Persyaratan dokumen:**
        - Format PDF (bukan gambar/scan)
        - Setiap soal diberi nomor `1.` `2.` dst.
        - Mendukung pilihan ganda dan esai
        """)


# =============================================================================
# KARTU: TOMBOL & KONTROL ANALISIS
# =============================================================================

with st.container(border=True):
    st.markdown('<p class="card-title">Analisis</p>', unsafe_allow_html=True)

    btn_col, note_col = st.columns([2, 5], gap="medium")
    with btn_col:
        analisis_clicked = st.button(
            "Mulai Analisis Dokumen",
            type="primary",
            disabled=(uploaded_file is None),
        )
    with note_col:
        if uploaded_file is None:
            st.warning("Unggah file PDF terlebih dahulu untuk mengaktifkan tombol analisis.")


# =============================================================================
# PIPELINE UTAMA
# =============================================================================

if analisis_clicked and uploaded_file is not None:
    import time
    start_time = time.time()
    output_path = str(ROOT_DIR / "data" / "output" / "hasil_anotasi.pdf")
    (ROOT_DIR / "data" / "output").mkdir(parents=True, exist_ok=True)

    # --- Tahap 1: Ekstraksi Soal ---
    with st.status("Tahap 1/3 — Mengekstrak butir soal dari PDF…", expanded=True) as s1:
        try:
            st.write("Menyimpan file ke direktori sementara…")
            input_path = simpan_file_upload(uploaded_file)
            st.write(f"Tersimpan: `{input_path}`")
        except IOError as e:
            st.error(f"Gagal menyimpan file: {e}")
            st.stop()

        try:
            processor = PDFProcessor()
            questions: list[str] = processor.process_document(input_path)
        except Exception as e:
            st.error(f"Gagal membaca PDF: {e}")
            st.stop()

        if not questions:
            st.error("Tidak ada butir soal yang berhasil diekstrak. Periksa format dokumen.")
            st.stop()

        s1.update(
            label=f"Tahap 1/3 selesai — {len(questions)} butir soal diekstrak.",
            state="complete",
        )

    # --- Tahap 2: Klasifikasi LLM (Batch Mode) ---
    BATCH_SIZE = 10
    total_chunks = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE

    with st.status(
        f"Tahap 2/3 — Mengevaluasi {len(questions)} soal "
        f"dalam {total_chunks} batch (maks. {BATCH_SIZE} soal/request)…",
        expanded=True,
    ) as s2:
        try:
            st.write("Menginisialisasi CloudLLMRouter (Gemini Cascading)…")
            router = CloudLLMRouter()
        except Exception as e:
            st.error(f"Gagal menginisialisasi LLM Router: {e}")
            st.stop()
            
        st.write(
            f"Strategi: **Gemini Cascading Fallback** — "
            f"{len(questions)} soal diproses dalam "
            f"chunk per {BATCH_SIZE} soal per request API. "
            f"Model aktif: `{', '.join(router._gemini_client and ['Terhubung'] or ['Offline'])}`"
        )

        prog = st.progress(0, text="Memulai batch evaluasi…")
        
        evaluation_results: list[dict] = []
        
        # Eksekusi evaluate_batch kustom dengan progress sinkron. Karena evaluate_batch 
        # dari llm_router tidak melempar callback, kita jalankan manual loop batch 
        # di aplikasi agar progress bar ter-update real-time.
        for chunk_idx in range(total_chunks):
            start = chunk_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(questions))
            chunk = questions[start:end]
            
            prog.progress(
                int((chunk_idx / total_chunks) * 100),
                text=f"Memproses Batch {chunk_idx + 1}/{total_chunks} (Soal {start+1}–{end})…"
            )
            
            # Panggil kaskade Gemini untuk chunk ini
            batch_res = router._call_gemini_cascade(chunk)
            
            # Tampilkan model yang berhasil merespons
            if router.active_model and not all("error" in r for r in batch_res):
                st.write(f"  Batch {chunk_idx+1}: Diproses via `{router.active_model}`")
            elif all("error" in r for r in batch_res):
                st.warning(f"  Batch {chunk_idx+1}: Seluruh rantai model Gemini gagal merespons.")
            
            # --- PROTEKSI HALUSINASI LLM & AKURASI ANOTASI ---
            # Jika model (khususnya Ollama) menghapus/melewatkan soal, pad dengan ERROR
            while len(batch_res) < len(chunk):
                batch_res.append({
                    "kategori": "ERROR",
                    "error": "LLM halusinasi dan menghilangkan soal ini",
                    "is_question": True,
                    "model_used": "System-Fallback"
                })
                
            # PAKSA timpa teks soal dengan teks mentah ASLI dari PDF.
            # Ini sangat krusial agar PyMuPDF di modul visual_annotator bisa
            # menemukan koordinat anotasi secara presisi 100% tanpa gangguan modifikasi dari LLM.
            for i, res in enumerate(batch_res):
                if i < len(chunk):
                    res["question_text"] = chunk[i]
                    
            evaluation_results.extend(batch_res)
            
            # Memberi JEDA agar terhindar dari Error 429
            if chunk_idx < total_chunks - 1:
                import time as _time
                _time.sleep(3.0)

        prog.progress(100, text="Evaluasi batch selesai.")
        
        s2.update(
            label=f"Tahap 2/3 selesai — {len(evaluation_results)} soal dievaluasi.",
            state="complete",
        )

    # --- Short-Circuit di Front-End ---
    # Periksa apakah seluruh hasil balasan berbentuk error atau tidak ada kategori valid
    has_valid_result = any(
        "kategori" in r and str(r.get("kategori")).upper() not in ["ERROR", "N/A", "", "NONE"] 
        for r in evaluation_results
    )
    if not has_valid_result:
        st.error(
            "**Evaluasi Dihentikan:** Seluruh rantai model Gemini (Cloud) "
            "tidak dapat dijangkau atau mengembalikan data kosong. "
            "Silakan periksa koneksi internet dan API Key, "
            "lalu tunggu beberapa saat sebelum mencoba kembali."
        )
        st.stop()

    # --- Tahap 3: Anotasi PDF ---
    with st.status("Tahap 3/3 — Membuat anotasi warna pada PDF…", expanded=True) as s3:
        annot_result = None
        try:
            annotator = PDFAnnotator()
            annot_result = annotator.annotate_document(
                input_pdf_path=input_path,
                output_pdf_path=output_path,
                evaluation_results=evaluation_results,
            )
            s3.update(
                label=(
                    f"Tahap 3/3 selesai — "
                    f"{annot_result.annotated_count}/{annot_result.total_questions} soal dianotasi."
                ),
                state="complete",
            )
        except Exception as e:
            st.warning(f"Anotasi PDF tidak berhasil: {e}")
            s3.update(label="Tahap 3/3 selesai (anotasi dilewati).", state="complete")

    # Simpan ke session state
    df_results = pd.DataFrame([
        {
            "No": idx + 1,
            "Butir Soal": r.get("question_text", "").strip(),
            "KKO": r.get("kko", "—") if r.get("is_question", True) else "Bukan Soal",
            "Level": r.get("level", "—") if r.get("is_question", True) else "-",
            "Kategori": r.get("kategori", "ERROR") if r.get("is_question", True) else "N/A",
            "Tipe Soal": r.get("tipe_soal", "Pilihan Ganda") if r.get("is_question", True) else "N/A",
            "Anomali Opsi": str(r.get("anomali_opsi", "Aman")).strip() if r.get("is_question", True) else "N/A",
            "Model": r.get("model_used", "—"),
        }
        for idx, r in enumerate(evaluation_results)
    ])

    end_time = time.time()
    duration = end_time - start_time
    st.session_state.update({
        "df_results":         df_results,
        "evaluation_results": evaluation_results,
        "output_path":        output_path,
        "annot_result":       annot_result,
        "filename":           uploaded_file.name,
        "duration":           duration,
    })
    st.success(f"Analisis selesai dalam waktu {duration:.1f} detik.")


# =============================================================================
# BLOK HASIL — DITAMPILKAN DARI SESSION STATE
# =============================================================================

if "df_results" in st.session_state:
    df: pd.DataFrame = st.session_state["df_results"]
    output_path: str = st.session_state["output_path"]
    annot_result     = st.session_state.get("annot_result")
    filename: str    = st.session_state.get("filename", "soal.pdf")
    stats            = hitung_statistik(df)

    st.divider()
    st.subheader("Hasil Analisis")

    # --- KARTU: METRIK DISTRIBUSI KOGNITIF ---
    with st.container(border=True):
        st.markdown('<p class="card-title">Ringkasan Distribusi Kognitif</p>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Soal", stats["total"])
        m2.metric("LOTS — C1",   f"{stats['LOTS']['count']} soal",  f"{stats['LOTS']['pct']}%",  delta_color="off")
        m3.metric("MOTS — C2 & C3",        f"{stats['MOTS']['count']} soal",  f"{stats['MOTS']['pct']}%",  delta_color="off")
        m4.metric("HOTS — C4, C5, C6",f"{stats['HOTS']['count']} soal",  f"{stats['HOTS']['pct']}%",  delta_color="off")

        st.markdown("---")

        hots_pct = stats["HOTS"]["pct"]
        if 20 <= hots_pct <= 30:
            st.success(
                f"**Proporsi HOTS: {hots_pct}%** — "
                "Instrumen asesmen memiliki distribusi HOTS yang ideal (20% - 30%)."
            )
        elif hots_pct > 30:
            st.warning(
                f"**Proporsi HOTS: {hots_pct}%** — "
                "Proporsi HOTS melebihi batas ideal (20% - 30%). Instrumen asesmen mungkin terlalu sulit bagi siswa."
            )
        else:
            st.error(
                f"**Proporsi HOTS: {hots_pct}%** — "
                "Proporsi HOTS di bawah batas ideal (20% - 30%). Disarankan untuk menambah soal tingkat analisis, evaluasi, atau mencipta."
            )

    # --- NAVIGASI TAB ---
    tab1, tab2 = st.tabs(["Dasbor Analitik", "Detail Klasifikasi & Unduh"])

    # =========================================================
    # TAB 1: DASBOR ANALITIK
    # =========================================================
    with tab1:

        col_bar, col_panel = st.columns([3, 2], gap="large")

        # Grafik distribusi kategori
        with col_bar:
            with st.container(border=True):
                st.markdown('<p class="card-title">Distribusi LOTS / MOTS / HOTS</p>', unsafe_allow_html=True)
                if stats["total"] > 0:
                    chart_df = pd.DataFrame({
                        "Kategori": ["LOTS (C1)", "MOTS (C2–C3)", "HOTS (C4–C6)"],
                        "Jumlah Soal": [
                            stats["LOTS"]["count"],
                            stats["MOTS"]["count"],
                            stats["HOTS"]["count"],
                        ],
                    }).set_index("Kategori")
                    st.bar_chart(chart_df, y="Jumlah Soal", height=260, width="stretch")
                else:
                    st.info("Tidak ada data.")

        # Panel info AI + anotasi
        with col_panel:


            if annot_result is not None:
                with st.container(border=True):
                    st.markdown('<p class="card-title">Hasil Anotasi PDF</p>', unsafe_allow_html=True)
                    st.markdown(
                        f"| Keterangan | Nilai |\n|---|---|\n"
                        f"| Soal teranotasi | **{annot_result.annotated_count}** |\n"
                        f"| Tidak ditemukan | **{annot_result.not_found_count}** |\n"
                        f"| Error/dilewati | **{annot_result.error_count}** |\n"
                        f"| Diabaikan (Bukan Soal) | **{getattr(annot_result, 'ignored_count', 0)}** |\n"
                        f"| Keberhasilan | **{annot_result.success_rate:.1f}%** |"
                    )
            else:
                st.info("Anotasi PDF tidak tersedia.")

        # Distribusi per level (C1–C6)
        with st.container(border=True):
            st.markdown('<p class="card-title">Distribusi per Level Bloom (C1–C6)</p>', unsafe_allow_html=True)
            if stats["total"] > 0:
                level_counts = (
                    df["Level"]
                    .value_counts()
                    .reindex(["C1", "C2", "C3", "C4", "C5", "C6"], fill_value=0)
                )
                level_df = pd.DataFrame({
                    "Level": level_counts.index,
                    "Jumlah": level_counts.values,
                }).set_index("Level")
                st.bar_chart(level_df, y="Jumlah", height=200, width="stretch")

    # =========================================================
    # TAB 2: DETAIL KLASIFIKASI & UNDUH
    # =========================================================
    with tab2:

        # Ringkasan cepat dengan warna semantik per kategori
        st.markdown("**Ringkasan Kategori**")
        qa, qb, qc = st.columns(3)
        with qa:
            if stats["LOTS"]["count"] > 0:
                st.success(
                    f"**LOTS — {stats['LOTS']['count']} soal ({stats['LOTS']['pct']}%)**\n\n"
                    "C1 Mengingat"
                )
        with qb:
            if stats["MOTS"]["count"] > 0:
                st.warning(
                    f"**MOTS — {stats['MOTS']['count']} soal ({stats['MOTS']['pct']}%)**\n\n"
                    "C2 Memahami & C3 Mengaplikasikan"
                )
        with qc:
            if stats["HOTS"]["count"] > 0:
                st.error(
                    f"**HOTS — {stats['HOTS']['count']} soal ({stats['HOTS']['pct']}%)**\n\n"
                    "C4 Menganalisis · C5 Mengevaluasi · C6 Mencipta"
                )
        if stats["ERROR"]["count"] > 0:
            st.info(
                f"**{stats['ERROR']['count']} soal** tidak dapat diklasifikasikan "
                "(seluruh rantai model Gemini gagal merespons pada saat pemrosesan)."
            )

        st.divider()

        # Tabel detail
        with st.container(border=True):
            st.markdown('<p class="card-title">Tabel Hasil Evaluasi per Butir Soal</p>', unsafe_allow_html=True)
            if len(df) > 0:
                
                # Check for anomalies to display warning
                anomali_df = df[(~df["Anomali Opsi"].str.upper().isin(["AMAN", "N/A", "-", "TIDAK RELEVAN (ESAI)"])) & (df["Tipe Soal"].str.upper() != "ESAI")]
                for _, row in anomali_df.iterrows():
                    st.warning(f"**Soal No. {row['No']}** terdeteksi anomali: {row['Anomali Opsi']}")

                # Gunakan st.table agar teks bisa melakukan text-wrap ke bawah (tidak terpotong).
                # Set index ke "No" supaya tabel tidak menampilkan index default Pandas (0, 1, 2, dst.)
                df_table = df.set_index("No")

                # Styling anomaly rows
                def highlight_anomali(s):
                    if str(s.get("Tipe Soal", "")).upper() == "ESAI":
                        return ['' for _ in s]
                    is_anomali = str(s.get("Anomali Opsi", "")).upper() not in ["AMAN", "N/A", "-", "TIDAK RELEVAN (ESAI)"]
                    return ['background-color: rgba(255, 100, 100, 0.2)' if is_anomali else '' for _ in s]

                try:
                    styled_df = df_table.style.apply(highlight_anomali, axis=1)
                except Exception:
                    styled_df = df_table # Fallback if styling fails

                # Menampilkan dengan st.table supaya sel dapat diperluas ke bawah (wrap)
                st.table(styled_df)
            else:
                st.warning("Tidak ada data untuk ditampilkan.")

        st.divider()

        # Tombol unduh
        with st.container(border=True):
            st.markdown('<p class="card-title">Unduh Hasil</p>', unsafe_allow_html=True)
            dl1, dl2, _ = st.columns([2, 2, 4])

            with dl1:
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    with open(output_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="Unduh PDF Teranotasi",
                        data=pdf_bytes,
                        file_name=f"Anotasi_Bloom_{filename}",
                        mime="application/pdf",
                        type="primary",
                    )
                else:
                    st.warning("File PDF hasil anotasi tidak tersedia.")

            with dl2:
                if len(df) > 0:
                    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        label="Unduh Ringkasan CSV",
                        data=csv_bytes,
                        file_name=f"Klasifikasi_Bloom_{filename.replace('.pdf', '')}.csv",
                        mime="text/csv",
                    )



# =============================================================================
# TAMPILAN AWAL (belum ada data)
# =============================================================================
elif uploaded_file is None and "df_results" not in st.session_state:
    with st.container(border=True):
        st.info(
            "**Belum ada dokumen yang dianalisis.**\n\n"
            "Unggah file PDF soal ujian di panel atas, lalu klik "
            "**Mulai Analisis Dokumen** untuk memulai."
        )
