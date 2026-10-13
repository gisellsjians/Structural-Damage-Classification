"""Prediksi data uji dan pembuatan berkas submission untuk Structural Damage Classification.
Modul ini memuat model terlatih dari folder ``models/`` (hasil ``scripts/train.py``),
membuat prediksi untuk ``data/test.csv``, lalu menulis ``submissions/submission.csv``.

Alur::
    models/pipeline_model_<nama>.pkl  --\\
    data/test.csv                     ----> predict_in_chunks -> build_submission
    data/sample_submission.csv        --/                       -> validate_submission
                                                                -> submissions/submission.csv

Contoh penggunaan dari terminal (dijalankan di root repositori)::
    python -m scripts.predict
    python -m scripts.predict --model-name random_forest
    python -m scripts.predict --model-path models/pipeline_model_reglog.pkl --output submissions/submission_reglog.csv

Tanpa argumen, model dipilih dari ``models/best_params.json`` (ditulis oleh
``notebooks/02_modeling.ipynb``); jika berkas itu tidak ada, dipakai satu-satunya
berkas ``pipeline_model_*.pkl`` yang tersedia.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

# Pastikan root repositori ada di sys.path agar impor ``scripts.*`` konsisten, baik dijalankan sebagai modul maupun dari notebook.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess import ID_COL, TARGET  # noqa: E402
from scripts.train import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, load_model, load_params  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SUBMISSION_PATH = ROOT / "submissions" / "submission.csv"


# model
def resolve_model_path(models_dir: str | Path = DEFAULT_MODELS_DIR,
                       model_name: str | None = None) -> Path:
    """Tentukan berkas model yang akan dipakai.
    Urutan penentuan:
        1. ``model_name`` diberikan -> ``models/pipeline_model_<model_name>.pkl``.
        2. ``models/best_params.json`` ada -> nama model pada berkas itu.
        3. Hanya ada satu berkas ``pipeline_model_*.pkl`` -> berkas tersebut.
    
    Args:
        models_dir: Folder tempat model disimpan.
        model_name: Kunci model (mis. ``"random_forest"``); ``None`` untuk otomatis.
    Returns:
        Path berkas ``.pkl`` yang ditemukan.
    Raises:
        FileNotFoundError: Jika berkas tidak ditemukan.
        ValueError: Jika ada beberapa model dan tidak ada petunjuk model mana yang dipakai.
    """
    models_dir = Path(models_dir)
    if model_name is None:
        params_file = models_dir / "best_params.json"
        if params_file.exists():
            model_name, _ = load_params(params_file)
    if model_name is not None:
        path = models_dir / f"pipeline_model_{model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Berkas model tidak ditemukan: {path}")
        return path

    candidates = sorted(models_dir.glob("pipeline_model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada berkas pipeline_model_*.pkl di {models_dir}. "
                                "Latih model terlebih dahulu (python -m scripts.train).")
    if len(candidates) > 1:
        raise ValueError(f"Ditemukan beberapa model {[c.name for c in candidates]}. "
                         "Tentukan salah satu dengan --model-name atau --model-path.")
    return candidates[0]


def check_versions(model_path: str | Path) -> list[str]:
    """Bandingkan versi scikit-learn saat ini dengan versi saat model dilatih.
    Berkas ``.pkl`` terikat pada versi pustaka, ketidakcocokan dapat menyebabkan
    kesalahan atau hasil yang berbeda. Versi saat pelatihan dibaca dari berkas
    ``pipeline_model_<nama>_meta.json`` yang berada di folder yang sama.
    
    Args:
        model_path: Lokasi berkas model ``.pkl``.
    Returns:
        Daftar pesan peringatan (kosong jika versi cocok atau berkas meta tidak ada).
    """
    model_path = Path(model_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    if not meta_path.exists():
        return []
    trained = json.loads(meta_path.read_text(encoding="utf-8")).get("versions", {}).get("scikit-learn")
    if trained and trained != sklearn.__version__:
        return [f"Versi scikit-learn berbeda: model dilatih dengan {trained}, "
                f"lingkungan saat ini {sklearn.__version__}."]
    return []


# prediksi
def predict_in_chunks(model: Pipeline, X: pd.DataFrame, chunk_size: int = 50_000) -> pd.DataFrame:
    """Buat prediksi per potongan data agar pemakaian memori terkendali.
    Jika model mendukung ``predict_proba``, label dipilih sebagai kelas dengan
    probabilitas tertinggi (setara dengan ``predict`` untuk model yang dipakai pada
    proyek ini) dan kolom ``confidence`` berisi probabilitas tertinggi tersebut.
    Jika tidak (mis. One-vs-One), dipakai ``predict`` dan ``confidence`` berisi NaN.
    
    Args:
        model: Pipeline terlatih (preprocessing + model).
        X: Data uji mentah (termasuk ``building_id``; dibuang oleh ``DataCleaner``).
        chunk_size: Jumlah baris per potongan.
    Returns:
        DataFrame berindeks sama dengan ``X`` dengan kolom ``damage_grade``,
        ``confidence``, dan satu kolom probabilitas per kelas (``p_<kelas>``)
        bila tersedia.
    """
    use_proba = hasattr(model, "predict_proba")
    classes = np.asarray(model.classes_)
    labels, probas = [], []
    for start in range(0, len(X), chunk_size):
        chunk = X.iloc[start:start + chunk_size]
        if use_proba:
            p = model.predict_proba(chunk)
            probas.append(p)
            labels.append(classes[p.argmax(axis=1)])
        else:
            labels.append(np.asarray(model.predict(chunk)))

    out = pd.DataFrame({TARGET: np.concatenate(labels)}, index=X.index)
    if use_proba:
        proba = np.vstack(probas)
        out["confidence"] = proba.max(axis=1)
        for i, cls in enumerate(classes):
            out[f"p_{cls}"] = proba[:, i]
    else:
        out["confidence"] = np.nan
    return out


# submission
def build_submission(test: pd.DataFrame, labels, sample_submission: pd.DataFrame) -> pd.DataFrame:
    """Susun DataFrame submission dengan urutan baris dan kolom seperti ``sample_submission``.
    Prediksi dicocokkan berdasarkan ``building_id`` (bukan berdasarkan urutan baris),
    karena urutan baris pada ``test.csv`` dan ``sample_submission.csv`` dapat berbeda.

    Args:
        test: DataFrame uji yang memuat ``building_id``.
        labels: Label prediksi, sejajar dengan baris ``test``.
        sample_submission: Berkas contoh submission.
    Returns:
        DataFrame dengan kolom yang sama dengan ``sample_submission``.
    Raises:
        ValueError: Jika ``building_id`` pada test tidak unik, atau ada
            ``building_id`` pada ``sample_submission`` yang tidak memiliki prediksi.
    """
    if not test[ID_COL].is_unique:
        raise ValueError("building_id pada data uji tidak unik.")
    predictions = pd.Series(np.asarray(labels), index=test[ID_COL].to_numpy())

    submission = sample_submission[[ID_COL]].copy()
    submission[TARGET] = submission[ID_COL].map(predictions)
    missing = int(submission[TARGET].isna().sum())
    if missing:
        raise ValueError(f"{missing} building_id pada sample_submission tidak memiliki prediksi.")
    return submission[list(sample_submission.columns)]


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame,
                        valid_labels: list[str]) -> dict[str, bool]:
    """Periksa bahwa submission sesuai dengan format yang diminta panitia.
    Pemeriksaan: kolom identik, jumlah baris sama, ``building_id`` unik dan identik
    (beserta urutannya) dengan ``sample_submission``, tidak ada nilai kosong, dan
    semua label termasuk dalam ``valid_labels``.

    Args:
        submission: DataFrame submission yang akan diperiksa.
        sample_submission: Berkas contoh dari panitia.
        valid_labels: Label yang sah (mis. ``["Grade 1", ..., "Grade 5"]``).
    Returns:
        Dict nama pemeriksaan -> ``True`` untuk seluruh pemeriksaan yang lulus.
    Raises:
        ValueError: Jika ada pemeriksaan yang gagal (pesan memuat semua kegagalan).
    """
    same_columns = list(submission.columns) == list(sample_submission.columns)
    same_rows = len(submission) == len(sample_submission)
    checks = {
        "kolom identik dengan sample_submission": same_columns,
        "jumlah baris sama": same_rows,
        "building_id unik": bool(submission[ID_COL].is_unique) if ID_COL in submission else False,
        "building_id & urutan identik": (same_rows and ID_COL in submission
                                         and bool((submission[ID_COL].to_numpy()
                                                   == sample_submission[ID_COL].to_numpy()).all())),
        "tidak ada nilai kosong": bool(submission.notna().all().all()),
        "seluruh label sah": (TARGET in submission
                              and bool(submission[TARGET].isin(valid_labels).all())),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Validasi submission gagal: {failed}")
    return checks


def save_submission(submission: pd.DataFrame, path: str | Path = DEFAULT_SUBMISSION_PATH) -> Path:
    """Tulis submission ke CSV tanpa kolom indeks.
    Args:
        submission: DataFrame submission yang sudah divalidasi.
        path: Lokasi berkas keluaran.
    Returns:
        Path berkas yang ditulis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path


def run_prediction(model_path: str | Path | None = None, model_name: str | None = None,
                   models_dir: str | Path = DEFAULT_MODELS_DIR,
                   data_dir: str | Path = DEFAULT_DATA_DIR,
                   output_path: str | Path = DEFAULT_SUBMISSION_PATH,
                   chunk_size: int = 50_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan seluruh alur: muat model, prediksi, susun, validasi, dan simpan submission.
    Args:
        model_path: Lokasi berkas ``.pkl``; jika ``None`` ditentukan oleh :func:`resolve_model_path`.
        model_name: Kunci model bila ``model_path`` tidak diberikan.
        models_dir: Folder model.
        data_dir: Folder yang berisi ``test.csv`` dan ``sample_submission.csv``.
        output_path: Lokasi ``submission.csv``.
        chunk_size: Ukuran potongan untuk prediksi.
    Returns:
        Tuple ``(submission, predictions)``; ``predictions`` adalah keluaran
        :func:`predict_in_chunks` (termasuk ``confidence``) berindeks sama dengan ``test.csv``.
    """
    model_path = Path(model_path) if model_path else resolve_model_path(models_dir, model_name)
    for message in check_versions(model_path):
        logger.warning(message)

    logger.info("Memuat model: %s", model_path)
    model = load_model(model_path)

    data_dir = Path(data_dir)
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    logger.info("Data uji: %s baris", f"{len(test):,}")

    started = time.perf_counter()
    predictions = predict_in_chunks(model, test, chunk_size=chunk_size)
    logger.info("Prediksi selesai dalam %.1f detik.", time.perf_counter() - started)

    submission = build_submission(test, predictions[TARGET], sample)
    validate_submission(submission, sample, [str(c) for c in model.classes_])
    saved = save_submission(submission, output_path)
    logger.info("Submission tersimpan di %s (%s baris)", saved, f"{len(submission):,}")
    return submission, predictions


# CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Baca argumen baris perintah.
    Args:
        argv: Daftar argumen; ``None`` berarti memakai ``sys.argv``.
    Returns:
        Namespace hasil ``argparse``.
    """
    parser = argparse.ArgumentParser(description="Buat prediksi data uji dan submission.csv.")
    parser.add_argument("--model-path", type=Path, help="Lokasi berkas model .pkl.")
    parser.add_argument("--model-name", help="Kunci model (mis. random_forest).")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Titik masuk CLI.
    Args:
        argv: Daftar argumen; ``None`` berarti memakai ``sys.argv``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    run_prediction(model_path=args.model_path, model_name=args.model_name,
                   models_dir=args.models_dir, data_dir=args.data_dir,
                   output_path=args.output, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()