"""Rekayasa fitur (feature engineering) untuk Structural Damage Classification.
Transformer ini *stateless* (tidak belajar apa pun dari data), sehingga aman
dipakai di dalam ``Pipeline`` dan validasi silang tanpa risiko data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Urutan kolom baru. get_feature_names_out() dan transform() memakai urutan yang sama.
ENGINEERED_BIN_COLS: list[str] = ["age_is_unknown", "age_is_new"]
ENGINEERED_NUM_COLS: list[str] = [
    "log_plinth_area",
    "area_per_floor",
    "height_per_floor",
    "n_rc_elements",
    "n_mud_elements",
]
NEW_COLUMNS: list[str] = ENGINEERED_BIN_COLS + ENGINEERED_NUM_COLS

# Kolom yang diperiksa untuk menghitung elemen RC / lumpur.
RC_SOURCE_COLS: list[str] = ["foundation_type", "roof_type", "ground_floor_type", "other_floor_type"]
MUD_SOURCE_COLS: list[str] = ["foundation_type", "ground_floor_type", "other_floor_type", "superstructure"]

REQUIRED_COLS: list[str] = sorted(
    {"count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq"}
    | set(RC_SOURCE_COLS)
    | set(MUD_SOURCE_COLS)
)


def count_rc_elements(df: pd.DataFrame) -> pd.Series:
    """Hitung berapa elemen bangunan yang berbahan beton bertulang (RC).

    Elemen yang diperiksa: fondasi, atap, lantai dasar, dan lantai lain. Nilai
    kategori RC pada kolom-kolom tersebut berawalan "RC" (``RC`` atau
    ``RCC/RB/RBC``), sehingga pencocokan memakai awalan huruf besar dan tahan
    terhadap perbedaan huruf kapital atau spasi.
    """
    flags = [
        df[col].astype(str).str.strip().str.upper().str.startswith("RC")
        for col in RC_SOURCE_COLS
    ]
    return pd.concat(flags, axis=1).sum(axis=1).astype(int)


def count_mud_elements(df: pd.DataFrame) -> pd.Series:
    """Hitung berapa elemen bangunan yang berbahan lumpur (mud).

    Elemen yang diperiksa: fondasi, lantai dasar, lantai lain, dan
    superstruktur. Pencocokan memakai kata "mud" tanpa membedakan huruf besar
    atau kecil, sehingga mencakup antara lain ``Mud mortar-Stone/Brick``,
    ``Mud``, ``Timber/Bamboo-Mud``, ``has_superstructure_mud_mortar_stone`` dan
    ``has_superstructure_adobe_mud``.
    """
    flags = [
        df[col].astype(str).str.contains("mud", case=False, regex=False)
        for col in MUD_SOURCE_COLS
    ]
    return pd.concat(flags, axis=1).sum(axis=1).astype(int)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Membuat fitur turunan dari kolom mentah yang sudah dibersihkan.

    Dipakai **setelah** ``DataCleaner`` (di ``scripts/preprocess.py``), yang
    mengubah ``age_building == 999`` menjadi ``NaN``. Fitur baru:
    ============================  ===============================================
    Fitur                         Arti
    ============================  ===============================================
    ``age_is_unknown``            1 jika umur bangunan tidak diketahui (NaN)
    ``age_is_new``                1 jika ``age_building == 0`` (bangunan baru)
    ``log_plinth_area``           ``log(1 + plinth_area_sq_ft)``
    ``area_per_floor``            luas dasar dibagi jumlah lantai
    ``height_per_floor``          tinggi bangunan dibagi jumlah lantai
    ``n_rc_elements``             jumlah elemen berbahan RC (0-4)
    ``n_mud_elements``            jumlah elemen berbahan lumpur (0-4)
    ============================  ===============================================
    Kolom asli tidak dihapus; pemilihan kolom akhir dilakukan oleh
    ``ColumnTransformer`` di ``scripts/preprocess.py``.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Validasi kolom input. Transformer ini tidak mempelajari apa pun.
        """
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan fitur turunan ke salinan ``X``.
        """
        self._check_columns(X)
        df = X.copy()

        # Jumlah lantai minimal 1 agar pembagian tidak pernah membagi dengan nol.
        floors = df["count_floors_pre_eq"].clip(lower=1)

        df["age_is_unknown"] = df["age_building"].isna().astype(int)
        df["age_is_new"] = (df["age_building"] == 0).astype(int)
        df["log_plinth_area"] = np.log1p(df["plinth_area_sq_ft"])
        df["area_per_floor"] = df["plinth_area_sq_ft"] / floors
        df["height_per_floor"] = df["height_ft_pre_eq"] / floors
        df["n_rc_elements"] = count_rc_elements(df)
        df["n_mud_elements"] = count_mud_elements(df)
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nama kolom keluaran: kolom input diikuti kolom baru.
        """
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + NEW_COLUMNS, dtype=object)

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        """Pastikan semua kolom wajib tersedia.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineer membutuhkan pandas DataFrame sebagai input.")
        missing = [c for c in REQUIRED_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")