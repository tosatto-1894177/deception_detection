"""
OpenFace Feature Loader
Carica e preprocessa le features OpenFace da CSV per deception detection

Features estratte:
- Action Units (35): 17 intensity + 18 presence
- Gaze (8): direzione sguardo 3D per entrambi gli occhi
- Pose (6): traslazione e rotazione testa
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional


class OpenFaceLoader:
    """
    Carica e preprocessa le features OpenFace da CSV
    """

    def __init__(self,
                 csv_dir: str,
                 use_aus: bool = True,
                 use_gaze: bool = True,
                 use_pose: bool = True,
                 normalize: bool = True,
                 verbose: bool = True):
        """
        Args:
            csv_dir: Cartella con i CSV OpenFace
            use_aus: Se True, carica Action Units (35 features)
            use_gaze: Se True, carica Gaze (8 features)
            use_pose: Se True, carica Pose (6 features)
            normalize: Se True, applica z-score normalization
            verbose: per gestire i print
        """
        self.csv_dir = Path(csv_dir)
        self.use_aus = use_aus
        self.use_gaze = use_gaze
        self.use_pose = use_pose
        self.normalize = normalize

        if not self.csv_dir.exists():
            raise FileNotFoundError(f"OpenFace CSV directory non trovata: {csv_dir}")

        # Definisce le colonne da estrarre
        self._define_feature_columns()

        # Statistiche per normalization
        self.feature_stats = None

        # Dimensione totale features
        self.feature_dim = self._compute_feature_dim()

        if verbose:
            print(f"OpenFaceLoader inizializzato:")
            print(f"  AUs:   {use_aus} ({'35 features' if use_aus else '0 features'})")
            print(f"  Gaze:  {use_gaze} ({'8 features' if use_gaze else '0 features'})")
            print(f"  Pose:  {use_pose} ({'6 features' if use_pose else '0 features'})")
            print(f"  Total: {self.feature_dim} features")
            print(f"  Normalize: {normalize}")

    def _define_feature_columns(self):
        """Definisce quali colonne estrarre dal CSV OpenFace"""

        # Action Units (17 intensity + 18 presence = 35 totali)
        self.au_intensity_cols = [
            ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
            ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
            ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
            ' AU26_r', ' AU45_r'
        ]

        self.au_presence_cols = [
            ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c',
            ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c',
            ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c',
            ' AU26_c', ' AU28_c', ' AU45_c'
        ]

        # Gaze (8 features: 3D direction per entrambi gli occhi + angoli)
        self.gaze_cols = [
            ' gaze_0_x', ' gaze_0_y', ' gaze_0_z',
            ' gaze_1_x', ' gaze_1_y', ' gaze_1_z',
            ' gaze_angle_x', ' gaze_angle_y'
        ]

        # Pose (6 features: translation + rotation)
        self.pose_cols = [
            ' pose_Tx', ' pose_Ty', ' pose_Tz',
            ' pose_Rx', ' pose_Ry', ' pose_Rz'
        ]

    def _compute_feature_dim(self) -> int:
        """Calcola dimensione totale features"""
        dim = 0
        if self.use_aus:
            dim += len(self.au_intensity_cols) + len(self.au_presence_cols)  # 35
        if self.use_gaze:
            dim += len(self.gaze_cols)  # 8
        if self.use_pose:
            dim += len(self.pose_cols)  # 6
        return dim

    def load_csv(self, clip_name: str) -> Optional[pd.DataFrame]:
        """
        Carica CSV OpenFace per una clip

        Args:
            clip_name: Nome della clip (es. 'AN_WILTY_EP15_lie4')

        Returns:
            DataFrame con features selezionate o None se non trovato
        """
        csv_path = self.csv_dir / f"{clip_name}.csv"

        if not csv_path.exists():
            print(f"CSV OpenFace non trovato: {clip_name}")
            return None

        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"❌ Errore caricamento CSV {clip_name}: {e}")
            return None

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estrae features selezionate dal DataFrame

        Args:
            df: DataFrame OpenFace completo

        Returns:
            Array (num_frames, feature_dim) con features selezionate
        """
        features_list = []

        # Action Units
        if self.use_aus:
            aus_intensity = df[self.au_intensity_cols].values
            aus_presence = df[self.au_presence_cols].values
            features_list.append(aus_intensity)
            features_list.append(aus_presence)

        # Gaze
        if self.use_gaze:
            gaze = df[self.gaze_cols].values
            features_list.append(gaze)

        # Pose
        if self.use_pose:
            pose = df[self.pose_cols].values
            features_list.append(pose)

        # Concatena tutto
        features = np.concatenate(features_list, axis=1)  # (num_frames, feature_dim)

        return features

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Applica z-score normalization

        Args:
            features: (num_frames, feature_dim)

        Returns:
            features normalizzate
        """
        if not self.normalize:
            return features

        # Z-score normalization: (x - mean) / std
        # Calcola su tutti i frame della clip
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)

        # Evita divisione per zero
        std = np.where(std == 0, 1.0, std)

        normalized = (features - mean) / std

        return normalized

    def load_and_preprocess(self, clip_name: str) -> Optional[torch.Tensor]:
        """
        Pipeline completa: carica CSV, estrae features, normalizza

        Args:
            clip_name: Nome della clip

        Returns:
            Tensor (num_frames, feature_dim) o None se errore
        """
        # 1. Carica CSV
        df = self.load_csv(clip_name)
        if df is None:
            return None

        # 2. Estrae features
        features = self.extract_features(df)

        # 3. Gestione NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Normalizzazione
        features = self.normalize_features(features)

        # 5. Converti a tensor
        features_tensor = torch.from_numpy(features).float()

        return features_tensor

    def get_feature_names(self) -> List[str]:
        """Ritorna lista nomi features"""
        names = []
        if self.use_aus:
            names.extend([col.strip() for col in self.au_intensity_cols])
            names.extend([col.strip() for col in self.au_presence_cols])
        if self.use_gaze:
            names.extend([col.strip() for col in self.gaze_cols])
        if self.use_pose:
            names.extend([col.strip() for col in self.pose_cols])
        return names