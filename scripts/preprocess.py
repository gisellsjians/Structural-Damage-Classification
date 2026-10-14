"""Pembersihan data dan preprocessing untuk Structural Damage Classification.
Alur lengkap (semuanya berada di dalam satu ``Pipeline`` scikit-learn):
    DataCleaner -> FeatureEngineer -> ColumnTransformer -> (model)
1. ``DataCleaner``       : buang ``building_id``, ubah ``age_building == 999`` jadi NaN,
                           jadikan kolom wilayah bertipe string (nominal).
2. ``FeatureEngineer``   : fitur turunan (lihat ``scripts/feature_engineering.py``).
3. ``ColumnTransformer`` : imputasi, clipping, scaling (numerik) dan encoding (kategorik).

Semua langkah yang *belajar dari data* (median, kuantil, rata-rata, kategori) hanya
dipelajari dari data latih saat ``fit``. Karena itu pipeline aman dipakai di dalam
validasi silang tanpa data leakage.

Contoh pemakaian::
    from scripts.preprocess import load_raw_data, split_features_target, make_model_pipeline
    from sklearn.linear_model import LogisticRegression
    train, test = load_raw_data("data")
    X, y = split_features_target(train)
    model = make_model_pipeline(LogisticRegression(max_iter=300, class_weight="balanced"))
    model.fit(X, y)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:  # dipakai sebagai package: from scripts.preprocess import ...
    from .feature_engineering import (ENGINEERED_BIN_COLS, ENGINEERED_NUM_COLS, FeatureEngineer,
                                      IDNeighborEncoder, neighbor_feature_names)
except ImportError:  # dijalankan langsung dari folder scripts/
    from feature_engineering import (ENGINEERED_BIN_COLS, ENGINEERED_NUM_COLS, FeatureEngineer,
                                     IDNeighborEncoder, neighbor_feature_names)

# konstanta
TARGET = "damage_grade"
ID_COL = "building_id"
AGE_SENTINEL = 999  # kode 'tidak diketahui' pada age_building (lihat EDA)

GEO_COLS = ["district_id", "vdcmun_id", "ward_id"]
RAW_NUM_COLS = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"]
CAT_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]

# Kolom yang masuk ke model. plinth_area_sq_ft digantikan versi log-nya
# (log_plinth_area) karena sebarannya berekor kanan panjang.
NUMERIC_FEATURES = ["count_floors_pre_eq", "age_building", "height_ft_pre_eq"] + ENGINEERED_NUM_COLS
BASE_NUMERIC_FEATURES = RAW_NUM_COLS  # tanpa fitur turunan (dipakai untuk ablation study)
BINARY_FEATURES = ENGINEERED_BIN_COLS
CATEGORICAL_FEATURES = CAT_COLS + ["district_id", "vdcmun_id"]  # 11 dan 110 level
HIGH_CARDINALITY_FEATURES = ["ward_id"]                        # 945 level


# data I/O
def load_raw_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baca ``train.csv`` dan ``test.csv`` dari folder data.
    Args:
        data_dir: Folder yang berisi ``train.csv`` dan ``test.csv``. Dari dalam
            folder ``notebooks/`` gunakan ``"../data"``.
    Returns:
        Tuple ``(train, test)`` berupa DataFrame.
    Raises:
        FileNotFoundError: Jika salah satu file tidak ada.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def split_features_target(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Pisahkan fitur dan target dari data latih.
    Args:
        train: DataFrame latih yang memuat kolom ``damage_grade``.
    Returns:
        Tuple ``(X, y)``. ``X`` berisi semua kolom selain target (``building_id``
        masih ada dan akan dibuang oleh ``DataCleaner``). ``y`` berisi label
        string ``"Grade 1"`` sampai ``"Grade 5"``.
    """
    return train.drop(columns=[TARGET]), train[TARGET]


# transformer
class DataCleaner(BaseEstimator, TransformerMixin):
    """Pembersihan dasar yang tidak bergantung pada data (stateless).
    Langkah:
        1. Buang ``building_id`` (identifier, bukan fitur) dan ``damage_grade``
           bila ikut terbawa.
        2. Ganti ``age_building == age_sentinel`` (999) menjadi ``NaN``.
           Nilai ini bukan umur nyata, sehingga akan diimputasi kemudian.
        3. Ubah ``district_id``, ``vdcmun_id``, ``ward_id`` menjadi string supaya
           diperlakukan sebagai kategori nominal, bukan angka.
    Args:
        age_sentinel: Kode 'tidak diketahui' pada ``age_building``.
    """

    def __init__(self, age_sentinel: int = AGE_SENTINEL):
        self.age_sentinel = age_sentinel

    def fit(self, X: pd.DataFrame, y=None) -> "DataCleaner":
        """Catat nama kolom input dan kolom keluaran.
        Args:
            X: DataFrame mentah.
            y: Diabaikan.
        Returns:
            ``self``.
        """
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_out_ = [c for c in X.columns if c not in (ID_COL, TARGET)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Terapkan pembersihan pada salinan ``X``.
        Args:
            X: DataFrame mentah (boleh masih memuat ``building_id``).
        Returns:
            DataFrame bersih tanpa ``building_id``/``damage_grade``.
        """
        df = X.drop(columns=[c for c in (ID_COL, TARGET) if c in X.columns]).copy()
        df["age_building"] = df["age_building"].mask(df["age_building"] == self.age_sentinel)
        for col in GEO_COLS:
            df[col] = df[col].astype(str)
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nama kolom keluaran (tanpa ``building_id`` dan target).
        Args:
            input_features: Diabaikan; nama diambil dari hasil ``fit``.
        Returns:
            Array nama kolom bertipe object.
        """
        return np.asarray(self.columns_out_, dtype=object)


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Potong nilai ekstrem (winsorizing) pada batas kuantil yang dipelajari saat fit.
    Pada data latih, batas bawah dan atas dihitung per kolom sebagai kuantil
    ``lower`` dan ``upper``. Nilai di luar batas ditarik ke batas tersebut,
    sehingga outlier (mis. tinggi bangunan >= 50 ft) tidak mendominasi model,
    terutama model linear. Batas hanya dipelajari dari data latih.
    Args:
        lower: Kuantil bawah, antara 0 dan 1.
        upper: Kuantil atas, antara 0 dan 1.
    """

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None) -> "QuantileClipper":
        """Hitung batas bawah dan atas tiap kolom dari data latih.
        Args:
            X: Array atau DataFrame numerik (tanpa NaN, setelah imputasi).
            y: Diabaikan.
        Returns:
            ``self``.
        Raises:
            ValueError: Jika ``lower`` tidak lebih kecil dari ``upper``.
        """
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Harus berlaku 0 <= lower < upper <= 1.")
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper, axis=0)
        self.n_features_in_ = X_arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X) -> np.ndarray:
        """Potong nilai di luar batas hasil ``fit``.
        Args:
            X: Array atau DataFrame numerik dengan jumlah kolom sama seperti saat fit.
        Returns:
            Array float dengan nilai terpotong pada batas kuantil.
        """
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


# pipeline
def build_preprocessor(
    scale: bool = True,
    encoding: str = "onehot",
    use_ward: bool = True,
    ward_min_frequency: int = 100,
    use_engineered: bool = True,
    use_id_neighbors: bool = False,
    neighbor_params: dict | None = None,
) -> ColumnTransformer:
    """Susun ``ColumnTransformer`` untuk fitur numerik, biner, dan kategorik.
    Args:
        scale: Jika ``True``, fitur numerik distandarisasi (rata-rata 0, simpangan
            baku 1). Perlu untuk model linear/berbasis jarak; tidak perlu untuk
            model berbasis pohon.
        encoding: ``"onehot"`` (cocok untuk model linear) atau ``"ordinal"``
            (cocok untuk model pohon; kolom lebih sedikit, keluaran dense).
        use_ward: Sertakan ``ward_id`` (945 level) atau tidak.
        ward_min_frequency: Pada mode ``"onehot"``, ward dengan jumlah baris
            latih kurang dari nilai ini digabung menjadi satu kategori 'jarang'.
        use_engineered: Jika ``False``, fitur turunan (``FeatureEngineer``) tidak
            dimasukkan ke model dan hanya empat kolom numerik mentah yang dipakai.
            Opsi ini digunakan untuk *ablation study*.
        use_id_neighbors: Jika ``True``, kolom fitur tetangga ``building_id`` (keluaran
            ``IDNeighborEncoder``) diteruskan ke model. Hanya berarti bila encoder tersebut
            ada pada pipeline (lihat :func:`build_preprocessing_pipeline`).
        neighbor_params: Argumen untuk ``IDNeighborEncoder`` (``windows``, ``prop_window``);
            ``None`` memakai nilai bawaan. Harus sama dengan yang dipakai pada
            :func:`build_preprocessing_pipeline` (dijamin bila lewat ``make_model_pipeline``).
    Returns:
        ``ColumnTransformer`` yang belum di-fit.
    Raises:
        ValueError: Jika ``encoding`` bukan ``"onehot"`` atau ``"ordinal"``.
    """
    if encoding not in {"onehot", "ordinal"}:
        raise ValueError("encoding harus 'onehot' atau 'ordinal'.")

    # Numerik: isi NaN dengan median -> potong outlier -> (opsional) standarisasi.
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper()),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # Kategorik: kategori yang tidak terlihat saat fit tidak menimbulkan kesalahan.
    if encoding == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore")
        ward_encoder = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=ward_min_frequency)
    else:
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ward_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    numeric_cols = NUMERIC_FEATURES if use_engineered else BASE_NUMERIC_FEATURES
    transformers = [
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_encoder, CATEGORICAL_FEATURES),
    ]
    if use_engineered:
        transformers.insert(1, ("bin", "passthrough", BINARY_FEATURES))
    if use_ward:
        transformers.append(("ward", ward_encoder, HIGH_CARDINALITY_FEATURES))
    if use_id_neighbors:
        transformers.append(("nbr", SimpleImputer(strategy="median"),
                             neighbor_feature_names(**(neighbor_params or {}))))

    # remainder="drop": kolom asli yang tidak terdaftar (mis. plinth_area_sq_ft) dibuang.
    return ColumnTransformer(transformers, remainder="drop")


def build_preprocessing_pipeline(**preprocessor_kwargs) -> Pipeline:
    """Susun pipeline preprocessing lengkap tanpa model.
    Args:
        **preprocessor_kwargs: Argumen untuk :func:`build_preprocessor`
            (``scale``, ``encoding``, ``use_ward``, ``ward_min_frequency``,
            ``use_engineered``, ``use_id_neighbors``, ``neighbor_params``).
    Returns:
        ``Pipeline`` dengan langkah ``clean`` -> ``features`` -> ``preprocess``.
        Bila ``use_id_neighbors=True``, langkah ``id_neighbors`` (``IDNeighborEncoder``)
        ditempatkan paling awal karena membutuhkan kolom ``building_id``.
    """
    steps = [
        ("clean", DataCleaner()),
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(**preprocessor_kwargs)),
    ]
    if preprocessor_kwargs.get("use_id_neighbors", False):
        steps.insert(0, ("id_neighbors", IDNeighborEncoder(**(preprocessor_kwargs.get("neighbor_params") or {}))))
    return Pipeline(steps)


def make_model_pipeline(estimator, **preprocessor_kwargs) -> Pipeline:
    """Gabungkan pipeline preprocessing dengan sebuah model.
    Args:
        estimator: Model scikit-learn (mis. ``LogisticRegression()``).
        **preprocessor_kwargs: Argumen untuk :func:`build_preprocessor`.
    Returns:
        ``Pipeline`` dengan langkah terakhir bernama ``model``.
    """
    pre = build_preprocessing_pipeline(**preprocessor_kwargs)
    return Pipeline([*pre.steps, ("model", estimator)])