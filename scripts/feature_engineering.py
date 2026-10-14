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


# ---------------------------------------------------------------------------
# Fitur tetangga berdasarkan urutan building_id (berbasis target, stateful)
# ---------------------------------------------------------------------------
NEIGHBOR_WINDOWS: tuple[int, ...] = (1, 3, 8, 25, 60, 150)
NEIGHBOR_PROP_WINDOW: int = 5
N_GRADES: int = 5


def neighbor_feature_names(windows: tuple[int, ...] = NEIGHBOR_WINDOWS,
                           prop_window: int = NEIGHBOR_PROP_WINDOW,
                           n_classes: int = N_GRADES) -> list[str]:
    """Nama kolom yang dihasilkan ``IDNeighborEncoder`` (urutannya sama dengan ``transform``).

    Args:
        windows: Ukuran jendela rata-rata (jumlah tetangga di tiap sisi).
        prop_window: Ukuran jendela untuk proporsi kelas.
        n_classes: Jumlah kelas target.

    Returns:
        Daftar nama kolom.
    """
    names = [f"nbr_mean_w{w}" for w in windows]
    names += [f"nbr_p{c + 1}_w{prop_window}" for c in range(n_classes)]
    names += ["nbr_prev_grade", "nbr_next_grade", "nbr_log_gap_prev", "nbr_log_gap_next"]
    return names


class IDNeighborEncoder(BaseEstimator, TransformerMixin):
    """Fitur dari label bangunan berlabel yang ``building_id``-nya berdekatan.

    ``building_id`` pada survei CBS dibentuk berurutan menurut wilayah dan waktu
    pendataan, sehingga bangunan dengan ID bersebelahan cenderung berada di
    lingkungan yang sama dan memiliki tingkat kerusakan yang mirip. Pada data
    latih, korelasi kerusakan dua bangunan berurutan masih sekitar 0,25 setelah
    rata-rata ward dikeluarkan. Transformer ini menangkap sinyal tersebut.

    Fitur:
        * ``nbr_mean_w{w}``       rata-rata grade ``2w`` tetangga berlabel terdekat
                                  (``w`` di kiri dan ``w`` di kanan menurut urutan ID).
        * ``nbr_p{c}_w{k}``       proporsi kelas ``c`` pada ``2k`` tetangga terdekat.
        * ``nbr_prev_grade`` / ``nbr_next_grade``  grade tetangga langsung.
        * ``nbr_log_gap_prev`` / ``nbr_log_gap_next``  log selisih ID ke tetangga langsung
                                  (proxy kerapatan pendataan).

    Pencegahan kebocoran (*leakage*): saat ``fit`` hanya label data latih yang
    disimpan. Baris yang ID-nya sama dengan salah satu ID latih (yaitu baris latih
    itu sendiri) **dikeluarkan dari tetangganya sendiri** (*leave-one-out*), sehingga
    labelnya tidak bocor ke fiturnya. Baris baru (validasi atau uji) memakai seluruh
    label latih. Karena langkah ini berada di dalam ``Pipeline``, validasi silang
    tetap bebas bocor: pada tiap fold hanya label fold latih yang dipakai.

    Catatan: fitur ini memanfaatkan struktur ID survei, bukan sifat fisik bangunan.
    Fitur ini opsional pada ``make_model`` (``use_id_neighbors``).

    Args:
        windows: Ukuran jendela rata-rata (tetangga per sisi).
        prop_window: Ukuran jendela proporsi kelas (tetangga per sisi).
        id_col: Nama kolom ID.
    """

    def __init__(self, windows: tuple[int, ...] = NEIGHBOR_WINDOWS,
                 prop_window: int = NEIGHBOR_PROP_WINDOW, id_col: str = "building_id"):
        self.windows = windows
        self.prop_window = prop_window
        self.id_col = id_col

    def fit(self, X: pd.DataFrame, y) -> "IDNeighborEncoder":
        """Simpan ID latih terurut beserta jumlah kumulatif label.

        Args:
            X: DataFrame latih yang memuat kolom ID.
            y: Label latih (string ``"Grade n"`` atau bilangan bulat). Urutan kelas
                diambil dari ``np.unique`` sehingga kelas terkecil menjadi grade 1.

        Returns:
            ``self``.
        """
        ids = np.asarray(X[self.id_col], dtype=np.int64)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        n_cls = len(self.classes_)
        codes = np.searchsorted(self.classes_, y_arr)

        order = np.argsort(ids, kind="stable")
        self.ids_, codes = ids[order], codes[order]
        onehot = np.zeros((len(codes), n_cls), dtype=np.float64)
        onehot[np.arange(len(codes)), codes] = 1.0
        self.grades_ = (codes + 1).astype(np.float64)
        self.cum_onehot_ = np.vstack([np.zeros((1, n_cls)), np.cumsum(onehot, axis=0)])
        self.cum_grade_ = np.concatenate([[0.0], np.cumsum(self.grades_)])
        self.global_mean_ = float(self.grades_.mean())
        return self

    def _window_sum(self, cum: np.ndarray, pos: np.ndarray, self_flag: np.ndarray, k: int):
        """Jumlah kumulatif pada jendela ``k`` tetangga per sisi, tanpa baris itu sendiri.

        Args:
            cum: Jumlah kumulatif (``n + 1`` baris; boleh berdimensi 2).
            pos: Posisi sisip tiap baris pada urutan ID latih.
            self_flag: 1 bila baris itu sendiri ada pada urutan (dikeluarkan), selain itu 0.
            k: Jumlah tetangga per sisi.

        Returns:
            Tuple ``(jumlah, banyak_tetangga)``.
        """
        n = len(self.ids_)
        lo = np.maximum(pos - k, 0)
        hi_start = pos + self_flag
        hi_end = np.minimum(hi_start + k, n)
        count = (pos - lo) + (hi_end - hi_start)
        total = cum[pos] - cum[lo] + cum[hi_end] - cum[hi_start]
        return total, count

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan fitur tetangga ke salinan ``X``.

        Args:
            X: DataFrame yang memuat kolom ID.

        Returns:
            Salinan ``X`` dengan kolom fitur tetangga (lihat
            :func:`neighbor_feature_names`) di sebelah kanan.
        """
        q = np.asarray(X[self.id_col], dtype=np.int64)
        n, n_cls = len(self.ids_), len(self.classes_)
        pos = np.searchsorted(self.ids_, q, side="left")
        is_self = (pos < n) & (self.ids_[np.minimum(pos, n - 1)] == q)
        self_flag = is_self.astype(np.int64)
        feats: dict[str, np.ndarray] = {}

        for w in self.windows:
            total, cnt = self._window_sum(self.cum_grade_, pos, self_flag, w)
            feats[f"nbr_mean_w{w}"] = np.where(cnt > 0, total / np.maximum(cnt, 1), self.global_mean_)

        c_sum, cnt = self._window_sum(self.cum_onehot_, pos, self_flag, self.prop_window)
        props = c_sum / np.maximum(cnt, 1)[:, None]
        for c in range(n_cls):
            feats[f"nbr_p{c + 1}_w{self.prop_window}"] = props[:, c]

        prev_idx, next_idx = pos - 1, pos + self_flag
        has_prev, has_next = prev_idx >= 0, next_idx < n
        feats["nbr_prev_grade"] = np.where(has_prev, self.grades_[np.clip(prev_idx, 0, n - 1)], self.global_mean_)
        feats["nbr_next_grade"] = np.where(has_next, self.grades_[np.clip(next_idx, 0, n - 1)], self.global_mean_)
        gap_prev = np.where(has_prev, q - self.ids_[np.clip(prev_idx, 0, n - 1)], 10 ** 12)
        gap_next = np.where(has_next, self.ids_[np.clip(next_idx, 0, n - 1)] - q, 10 ** 12)
        feats["nbr_log_gap_prev"] = np.log1p(gap_prev.astype(np.float64))
        feats["nbr_log_gap_next"] = np.log1p(gap_next.astype(np.float64))

        result = X.copy()
        for name in neighbor_feature_names(self.windows, self.prop_window, n_cls):
            result[name] = feats[name]
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Nama kolom keluaran: kolom input diikuti fitur tetangga.

        Args:
            input_features: Nama kolom input; bila ``None`` dipakai daftar kosong.

        Returns:
            Array nama kolom bertipe object.
        """
        base = list(input_features) if input_features is not None else []
        names = neighbor_feature_names(self.windows, self.prop_window, len(self.classes_))
        return np.asarray(base + names, dtype=object)
