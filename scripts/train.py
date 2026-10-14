"""Pelatihan dan penyimpanan model untuk Structural Damage Classification.
Isi utama:
    * ``MODEL_SPECS``      : daftar model kandidat (tanpa Neural Network) beserta
                             konfigurasi preprocessing dan ruang pencarian hiperparameter.
    * ``make_model``       : membangun ``Pipeline`` lengkap (preprocessing + model).
    * ``evaluate_cv``      : evaluasi validasi silang berstrata dengan metrik Macro F1.
    * ``train_and_save``   : melatih model pada seluruh data latih lalu menyimpannya
                             ke folder ``models/``.
    * ``tune_class_weights`` / ``fit_decision_weights`` : aturan keputusan yang menyetel
                             pengali probabilitas per kelas untuk memaksimalkan Macro F1.
    * ``load_model``       : memuat kembali model tersimpan tanpa pelatihan ulang.

Contoh penggunaan dari terminal (dijalankan di root repositori)::
    python -m scripts.train --list
    python -m scripts.train --model reglog
    python -m scripts.train --model hist_gb --cv
    python -m scripts.train --params-file models/best_params.json
    python -m scripts.train --params-file models/best_params.json --tune-weights

Berkas yang dihasilkan::
    models/pipeline_model_<nama_model>.pkl        pipeline terlatih (joblib)
    models/pipeline_model_<nama_model>_meta.json  catatan versi, parameter, dan waktu latih
    models/decision_weights.json                  pengali probabilitas per kelas (opsional)
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
    # HistGradientBoosting + fitur tetangga building_id (lihat IDNeighborEncoder). Berbeda dengan
    # kandidat lain, model ini memakai kolom building_id melalui encoder tersebut, bukan membuangnya.
    "hist_gb_nbr": ModelSpec(
        label="HistGradientBoosting + fitur tetangga ID",
        build_estimator=lambda: HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, n_iter_no_change=15, class_weight="balanced", random_state=SEED),
        preprocessor_kwargs={**_TREE_PREP, "use_id_neighbors": True},
        search_space={
            "model__learning_rate": loguniform(0.03, 0.2),
            "model__max_leaf_nodes": randint(15, 127),
            "model__min_samples_leaf": randint(20, 200),
            "model__l2_regularization": loguniform(1e-3, 10),
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


# --------------------------------------------------------------------------- aturan keputusan
DECISION_WEIGHTS_FILE = "decision_weights.json"


def _class_index(y, classes) -> np.ndarray:
    """Ubah label (string atau angka) menjadi indeks kelas ``0..K-1``.

    Args:
        y: Label sebenarnya.
        classes: Urutan kelas (mis. ``pipeline.classes_``).

    Returns:
        Array indeks bertipe int.
    """
    return np.searchsorted(np.asarray(classes), np.asarray(y))


def tune_class_weights(proba: np.ndarray, y_true_idx: np.ndarray, grid: np.ndarray | None = None,
                       n_rounds: int = 3) -> tuple[np.ndarray, float]:
    """Cari pengali probabilitas per kelas yang memaksimalkan Macro F1 (*coordinate ascent*).

    Prediksi akhir adalah ``argmax(proba * bobot)``. Bobot kelas pertama dikunci 1 karena
    hanya rasio antar-bobot yang berpengaruh. Untuk tiap kelas bergantian, bobotnya dicoba
    pada ``grid`` dan dipilih yang menaikkan Macro F1; proses diulang ``n_rounds`` putaran.

    Args:
        proba: Probabilitas kelas, bentuk ``(n, K)``, dari data yang **tidak** dipakai melatih model.
        y_true_idx: Indeks kelas sebenarnya, bentuk ``(n,)``.
        grid: Kandidat nilai bobot; bawaan 49 nilai eksponensial antara ``e^-1,2`` dan ``e^1,2``.
        n_rounds: Jumlah putaran.

    Returns:
        Tuple ``(bobot, macro_f1)`` dengan ``macro_f1`` dihitung pada data yang sama.
    """
    grid = np.exp(np.linspace(-1.2, 1.2, 49)) if grid is None else grid
    n_cls = proba.shape[1]
    w = np.ones(n_cls)
    best = f1_score(y_true_idx, (proba * w).argmax(1), average="macro")
    for _ in range(n_rounds):
        for c in range(1, n_cls):
            best_val = w[c]
            for g in grid:
                trial = w.copy()
                trial[c] = g
                score = f1_score(y_true_idx, (proba * trial).argmax(1), average="macro")
                if score > best + 1e-9:
                    best, best_val = score, g
            w[c] = best_val
    return w, float(best)


def apply_class_weights(proba: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pilih indeks kelas dengan ``argmax(proba * bobot)``.

    Args:
        proba: Probabilitas kelas, bentuk ``(n, K)``.
        weights: Pengali per kelas; ``None`` berarti ``argmax`` biasa.

    Returns:
        Array indeks kelas terpilih.
    """
    return (proba if weights is None else proba * np.asarray(weights)).argmax(axis=1)


def save_decision_weights(path: str | Path, model_name: str, classes, weights, info: dict) -> Path:
    """Simpan bobot keputusan beserta catatan evaluasinya ke JSON.

    Args:
        path: Lokasi berkas (mis. ``models/decision_weights.json``).
        model_name: Kunci model yang menghasilkan probabilitas.
        classes: Urutan kelas yang bobotnya sejajar.
        weights: Pengali per kelas.
        info: Catatan tambahan (skor sebelum/sesudah, ukuran data, dll.).

    Returns:
        Path berkas yang ditulis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "classes": [str(c) for c in classes],
               "weights": [float(v) for v in weights], **to_jsonable(info)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_decision_weights(path: str | Path, model_name: str | None = None) -> np.ndarray | None:
    """Baca bobot keputusan; ``None`` bila berkas tidak ada atau milik model lain.

    Args:
        path: Lokasi berkas JSON.
        model_name: Bila diberikan, bobot hanya dipakai jika cocok dengan nama model ini.

    Returns:
        Array bobot, atau ``None``.
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model_name is not None and payload.get("model") != model_name:
        return None
    return np.asarray(payload["weights"], dtype=float)


def fit_decision_weights(model_name: str, data_dir: str | Path = DEFAULT_DATA_DIR,
                         output_dir: str | Path = DEFAULT_MODELS_DIR, params: dict | None = None,
                         holdout_fraction: float = 0.20, seed: int = SEED,
                         n_jobs: int | None = None) -> dict:
    """Setel bobot keputusan pada data hold-out lalu simpan ke ``decision_weights.json``.

    Model dilatih pada ``1 - holdout_fraction`` data latih, probabilitas hold-out dipakai
    untuk menyetel bobot. Estimasi skor yang jujur dihitung dengan *cross-fitting*: bobot
    disetel pada separuh hold-out dan diuji pada separuh lainnya (dan sebaliknya). Bobot
    yang disimpan disetel pada seluruh hold-out.

    Args:
        model_name: Kunci model pada ``MODEL_SPECS`` (harus mendukung ``predict_proba``).
        data_dir: Folder ``train.csv``.
        output_dir: Folder tujuan ``decision_weights.json``.
        params: Hiperparameter tambahan untuk ``Pipeline.set_params``.
        holdout_fraction: Porsi data latih untuk hold-out.
        seed: Seed pengacakan.
        n_jobs: Jumlah proses paralel untuk estimator yang mendukungnya.

    Returns:
        Dict berisi ``weights``, ``macro_f1_argmax``, ``macro_f1_tuned_cross``, dan ``n_holdout``.
    """
    train, _ = load_raw_data(data_dir)
    X, y = split_features_target(train)
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=seed)

    pipe = make_model(model_name, n_jobs=n_jobs)
    if params:
        pipe.set_params(**params)
    pipe.fit(X_fit, y_fit)
    proba = pipe.predict_proba(X_hold)
    y_idx = _class_index(y_hold, pipe.classes_)

    idx = np.random.RandomState(seed).permutation(len(y_idx))
    half_a, half_b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    w_a, _ = tune_class_weights(proba[half_a], y_idx[half_a])
    w_b, _ = tune_class_weights(proba[half_b], y_idx[half_b])
    cross = np.mean([
        f1_score(y_idx[half_b], apply_class_weights(proba[half_b], w_a), average="macro"),
        f1_score(y_idx[half_a], apply_class_weights(proba[half_a], w_b), average="macro"),
    ])
    weights, in_sample = tune_class_weights(proba, y_idx)
    info = {
        "macro_f1_argmax": float(f1_score(y_idx, proba.argmax(1), average="macro")),
        "macro_f1_tuned_cross": float(cross),
        "macro_f1_tuned_in_sample": in_sample,
        "n_holdout": int(len(y_idx)),
        "holdout_fraction": holdout_fraction,
    }
    save_decision_weights(Path(output_dir) / DECISION_WEIGHTS_FILE, model_name, pipe.classes_, weights, info)
    return {"weights": weights, **info}


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
    parser.add_argument("--tune-weights", action="store_true",
                        help="Setel bobot keputusan per kelas (Macro F1) pada hold-out 20%% dan simpan.")
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

    if args.tune_weights:
        result = fit_decision_weights(model_name, args.data_dir, args.output_dir, params=params,
                                      n_jobs=args.n_jobs)
        logger.info("Bobot keputusan %s | Macro F1 hold-out: argmax %.4f -> tuned (cross-fit) %.4f",
                    np.round(result["weights"], 3), result["macro_f1_argmax"],
                    result["macro_f1_tuned_cross"])

    train_and_save(model_name, args.data_dir, args.output_dir, params=params,
                   cv_result=cv_result, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()