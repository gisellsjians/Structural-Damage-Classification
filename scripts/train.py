"""Pelatihan dan penyimpanan model untuk Structural Damage Classification.
Isi utama:
    * ``MODEL_SPECS``      : daftar model kandidat (tanpa Neural Network) beserta
                             konfigurasi preprocessing dan ruang pencarian hiperparameter.
    * ``make_model``       : membangun ``Pipeline`` lengkap (preprocessing + model).
    * ``evaluate_cv``      : evaluasi validasi silang berstrata dengan metrik Macro F1.
    * ``train_and_save``   : melatih model pada seluruh data latih lalu menyimpannya
                             ke folder ``models/``.
    * ``load_model``       : memuat kembali model tersimpan tanpa pelatihan ulang.

Contoh penggunaan dari terminal (dijalankan di root repositori)::
    python -m scripts.train --list
    python -m scripts.train --model reglog
    python -m scripts.train --model hist_gb --cv
    python -m scripts.train --params-file models/best_params.json

Berkas yang dihasilkan::
    models/pipeline_model_<nama_model>.pkl        pipeline terlatih (joblib)
    models/pipeline_model_<nama_model>_meta.json  catatan versi, parameter, dan waktu latih
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline

# Pastikan root repositori ada di sys.path agar impor ``scripts.*`` selalu konsisten,
# baik dijalankan sebagai modul (python -m scripts.train) maupun sebagai berkas.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess import load_raw_data, make_model_pipeline, split_features_target  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_MODELS_DIR = ROOT / "models"
GITHUB_LIMIT_MB = 100  # batas ukuran satu berkas pada GitHub


# --------------------------------------------------------------------------- registri model
@dataclass(frozen=True)
class ModelSpec:
    """Spesifikasi satu model kandidat.
    Attributes:
        label: Nama yang mudah dibaca untuk tabel dan grafik.
        build_estimator: Fungsi tanpa argumen yang mengembalikan estimator baru.
        preprocessor_kwargs: Argumen untuk ``build_preprocessor`` (scaling dan encoding).
        search_space: Ruang pencarian hiperparameter untuk ``RandomizedSearchCV``.
            Kunci memakai awalan ``model__`` karena estimator adalah langkah ``model``
            pada pipeline.
    """

    label: str
    build_estimator: Callable[[], Any]
    preprocessor_kwargs: dict
    search_space: dict


# Model linear memakai One-Hot + standardisasi, model pohon memakai Ordinal tanpa standardisasi (pohon tidak sensitif terhadap skala fitur).
_LINEAR_PREP = {"scale": True, "encoding": "onehot"}
_TREE_PREP = {"scale": False, "encoding": "ordinal"}


def _logreg() -> LogisticRegression:
    """Buat Logistic Regression biner/multinomial dengan bobot kelas seimbang."""
    return LogisticRegression(max_iter=300, class_weight="balanced")


MODEL_SPECS: dict[str, ModelSpec] = {
    "reglog": ModelSpec(
        label="Logistic Regression (multinomial)",
        build_estimator=_logreg,
        preprocessor_kwargs=_LINEAR_PREP,
        search_space={"model__C": loguniform(1e-2, 1e1)},
    ),
    "reglog_ovr": ModelSpec(
        label="Logistic Regression (One-vs-Rest)",
        build_estimator=lambda: OneVsRestClassifier(_logreg()),
        preprocessor_kwargs=_LINEAR_PREP,
        search_space={"model__estimator__C": loguniform(1e-2, 1e1)},
    ),
    "reglog_ovo": ModelSpec(
        label="Logistic Regression (One-vs-One)",
        build_estimator=lambda: OneVsOneClassifier(_logreg()),
        preprocessor_kwargs=_LINEAR_PREP,
        search_space={"model__estimator__C": loguniform(1e-2, 1e1)},
    ),

    # min_samples_leaf diberikan sebagai FRAKSI jumlah data latih (float), bukan angka tetap.
    # Ukuran pohon (dan ukuran berkas .pkl) kira-kira sebanding dengan 1 / fraksi dan tidak
    # bergantung pada jumlah baris, sehingga kapasitas yang disetel pada subsampel tetap
    # setara ketika model dilatih pada seluruh data.
    "random_forest": ModelSpec(
        label="Random Forest",
        build_estimator=lambda: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=1e-4, class_weight="balanced_subsample",
            n_jobs=1, random_state=SEED),
        preprocessor_kwargs=_TREE_PREP,
        search_space={
            "model__n_estimators": [100, 150],
            "model__max_depth": [None, 20, 30],
            "model__min_samples_leaf": [1e-4, 2e-4, 5e-4],
            "model__max_features": ["sqrt", 0.5],
        },
    ),
    "hist_gb": ModelSpec(
        label="HistGradientBoosting",
        build_estimator=lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=SEED),
        preprocessor_kwargs=_TREE_PREP,
        search_space={
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__max_leaf_nodes": randint(15, 80),
            "model__min_samples_leaf": randint(20, 200),
            "model__l2_regularization": loguniform(1e-3, 10),
        },
    ),
}


def list_models() -> list[str]:
    """Kembalikan nama-nama model kandidat yang tersedia.
    Returns:
        Daftar kunci ``MODEL_SPECS``.
    """
    return list(MODEL_SPECS)


def make_model(name: str, n_jobs: int | None = None, **preprocessor_overrides) -> Pipeline:
    """Bangun pipeline lengkap (preprocessing + model) untuk satu model kandidat.
    Args:
        name: Kunci model pada ``MODEL_SPECS`` (mis. ``"reglog"`` atau ``"hist_gb"``).
        n_jobs: Jumlah proses paralel untuk estimator yang mendukungnya (mis.
            Random Forest). ``None`` berarti memakai nilai bawaan estimator.
        **preprocessor_overrides: Menimpa argumen preprocessing bawaan model,
            mis. ``use_engineered=False`` untuk *ablation study*.
    Returns:
        ``Pipeline`` yang belum di-fit dengan langkah akhir bernama ``model``.
    Raises:
        KeyError: Jika ``name`` tidak terdaftar.
    """
    if name not in MODEL_SPECS:
        raise KeyError(f"Model '{name}' tidak dikenal. Pilihan: {list_models()}")
    spec = MODEL_SPECS[name]
    pipe = make_model_pipeline(spec.build_estimator(),
                               **{**spec.preprocessor_kwargs, **preprocessor_overrides})
    if n_jobs is not None and "n_jobs" in pipe.named_steps["model"].get_params():
        pipe.set_params(model__n_jobs=n_jobs)
    return pipe


# --------------------------------------------------------------------------- evaluasi
def get_cv(n_splits: int = 5, seed: int = SEED) -> StratifiedKFold:
    """Buat pembagi validasi silang berstrata yang dapat direproduksi.
    Args:
        n_splits: Jumlah fold.
        seed: Seed pengacakan.
    Returns:
        Objek ``StratifiedKFold`` dengan ``shuffle=True``.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                cv: StratifiedKFold | None = None, n_jobs: int = -1) -> dict:
    """Evaluasi pipeline dengan validasi silang berstrata (Macro F1 dan akurasi).
    Preprocessing berada di dalam pipeline sehingga dipelajari ulang pada setiap
    fold latih (tidak ada kebocoran data dari fold validasi).
    Args:
        pipeline: Pipeline yang belum di-fit.
        X: Fitur mentah (termasuk ``building_id``; dibuang oleh ``DataCleaner``).
        y: Target ``damage_grade``.
        cv: Pembagi validasi silang. Bawaan: ``get_cv()``.
        n_jobs: Jumlah proses paralel antar-fold.
    Returns:
        Dict berisi ``macro_f1_mean``, ``macro_f1_std``, ``accuracy_mean``,
        ``fit_seconds_mean`` dan ``fold_scores`` (Macro F1 tiap fold).
    """
    cv = cv or get_cv()
    res = cross_validate(pipeline, X, y, cv=cv, n_jobs=n_jobs, error_score="raise",
                         scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"})
    return {
        "macro_f1_mean": float(res["test_macro_f1"].mean()),
        "macro_f1_std": float(res["test_macro_f1"].std()),
        "accuracy_mean": float(res["test_accuracy"].mean()),
        "fit_seconds_mean": float(res["fit_time"].mean()),
        "fold_scores": [float(v) for v in res["test_macro_f1"]],
    }


# --------------------------------------------------------------------------- parameter & I/O
def to_jsonable(obj: Any) -> Any:
    """Ubah objek (termasuk tipe numpy) menjadi struktur yang dapat ditulis ke JSON.
    Args:
        obj: Dict, list, skalar Python, atau skalar numpy.
    Returns:
        Objek setara yang hanya berisi tipe dasar JSON.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_params(path: str | Path, model_name: str, params: dict,
                cv_macro_f1: float | None = None) -> Path:
    """Simpan hiperparameter terbaik ke berkas JSON.
    Args:
        path: Lokasi berkas JSON (mis. ``models/best_params.json``).
        model_name: Kunci model pada ``MODEL_SPECS``.
        params: Parameter pipeline (kunci berawalan ``model__``).
        cv_macro_f1: Skor Macro F1 validasi silang yang menyertai parameter (opsional).
    Returns:
        Path berkas yang ditulis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "params": to_jsonable(params), "cv_macro_f1": cv_macro_f1}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_params(path: str | Path) -> tuple[str, dict]:
    """Baca berkas parameter yang dibuat oleh :func:`save_params`.
    Args:
        path: Lokasi berkas JSON.
    Returns:
        Tuple ``(nama_model, params)``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["model"], payload["params"]


def load_model(path: str | Path) -> Pipeline:
    """Muat pipeline terlatih dari berkas ``.pkl``.
    Args:
        path: Lokasi berkas model hasil :func:`train_and_save`.
    Returns:
        Pipeline terlatih yang siap dipakai untuk ``predict``.
    """
    return joblib.load(path)


def train_and_save(model_name: str, data_dir: str | Path = DEFAULT_DATA_DIR,
                   output_dir: str | Path = DEFAULT_MODELS_DIR, params: dict | None = None,
                   cv_result: dict | None = None, n_jobs: int = -1) -> tuple[Pipeline, Path]:
    """Latih model pada seluruh data latih, lalu simpan pipeline dan catatannya.
    Args:
        model_name: Kunci model pada ``MODEL_SPECS``.
        data_dir: Folder yang berisi ``train.csv`` dan ``test.csv``.
        output_dir: Folder tujuan penyimpanan model.
        params: Hiperparameter tambahan untuk ``Pipeline.set_params`` (opsional).
        cv_result: Hasil :func:`evaluate_cv` yang ingin dicatat pada berkas meta (opsional).
        n_jobs: Jumlah proses paralel untuk estimator yang mendukungnya.
    Returns:
        Tuple ``(pipeline_terlatih, path_pkl)``.
    """
    train, _ = load_raw_data(data_dir)
    X, y = split_features_target(train)

    pipe = make_model(model_name, n_jobs=n_jobs)
    if params:
        pipe.set_params(**params)

    logger.info("Melatih '%s' pada %s baris x %s kolom ...", model_name, f"{len(X):,}", X.shape[1])
    started = time.perf_counter()
    pipe.fit(X, y)
    seconds = time.perf_counter() - started
    logger.info("Pelatihan selesai dalam %.1f detik.", seconds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"pipeline_model_{model_name}.pkl"
    joblib.dump(pipe, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6
    if size_mb > GITHUB_LIMIT_MB:
        logger.warning("Ukuran model %.0f MB melebihi batas GitHub (%d MB per berkas). Jangan di-commit; "
                       "tambahkan ke .gitignore atau kurangi kapasitas model.", size_mb, GITHUB_LIMIT_MB)

    meta = {
        "model_name": model_name,
        "label": MODEL_SPECS[model_name].label,
        "params": to_jsonable(params or {}),
        "n_train_rows": int(len(X)),
        "classes": [str(c) for c in pipe.classes_],
        "train_seconds": round(seconds, 1),
        "model_size_mb": round(size_mb, 1),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cv_result": to_jsonable(cv_result) if cv_result else None,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    meta_path = output_dir / f"pipeline_model_{model_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Model disimpan di %s", model_path)
    return pipe, model_path


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Baca argumen baris perintah.
    Args:
        argv: Daftar argumen; ``None`` berarti memakai ``sys.argv``.
    Returns:
        Namespace hasil ``argparse``.
    """
    parser = argparse.ArgumentParser(description="Latih dan simpan model Structural Damage Classification.")
    parser.add_argument("--model", choices=list_models(), help="Nama model yang dilatih.")
    parser.add_argument("--params-file", type=Path,
                        help="Berkas JSON hiperparameter (mis. models/best_params.json).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--cv", action="store_true", help="Jalankan validasi silang sebelum pelatihan akhir.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--list", action="store_true", help="Tampilkan daftar model lalu keluar.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Titik masuk CLI: (opsional) validasi silang, pelatihan akhir, dan penyimpanan.
    Args:
        argv: Daftar argumen; ``None`` berarti memakai ``sys.argv``.
    Raises:
        SystemExit: Jika argumen tidak valid.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)

    if args.list:
        for name, spec in MODEL_SPECS.items():
            print(f"{name:<15} {spec.label}")
        return

    params: dict = {}
    model_name = args.model
    if args.params_file:
        file_model, params = load_params(args.params_file)
        if model_name and model_name != file_model:
            raise SystemExit(f"--model '{model_name}' berbeda dengan model pada berkas parameter ('{file_model}').")
        model_name = file_model
    if not model_name:
        raise SystemExit("Tentukan --model atau --params-file (gunakan --list untuk melihat pilihan).")

    cv_result = None
    if args.cv:
        train, _ = load_raw_data(args.data_dir)
        X, y = split_features_target(train)
        pipe = make_model(model_name)
        if params:
            pipe.set_params(**params)
        cv_result = evaluate_cv(pipe, X, y, cv=get_cv(args.n_splits), n_jobs=args.n_jobs)
        logger.info("CV %d-fold: Macro F1 = %.4f (+/- %.4f)", args.n_splits,
                    cv_result["macro_f1_mean"], cv_result["macro_f1_std"])

    train_and_save(model_name, args.data_dir, args.output_dir, params=params,
                   cv_result=cv_result, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()