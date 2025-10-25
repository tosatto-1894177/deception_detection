import torch
import cv2
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class DOLOSFrameDataset(Dataset):
    """
    Dataset per DOLOS con annotazioni MUMIN.
    """

    def __init__(self, root_dir, annotation_file, transform=None, max_frames=None,
                 use_behavioral_features=False):
        """
        Args:
            root_dir: Directory con i frame estratti (organizzati per clip)
            annotation_file: Path al CSV con annotazioni MUMIN (train_dolos_example.csv)
            transform: Trasformazioni da applicare ai frame
            max_frames: Se specificato, tronca sequenze più lunghe (opzionale)
            use_behavioral_features: Se True, carica anche le feature comportamentali
                                     dal CSV (per uso futuro con OpenFace)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self.get_default_transform()
        self.max_frames = max_frames
        self.use_behavioral_features = use_behavioral_features

        # Carica annotazioni MUMIN
        self.annotations = self._load_annotations(annotation_file)

        # Trova tutte le clip con frame
        self.samples = []
        for clip_dir in sorted(self.root_dir.rglob("*")):
            if clip_dir.is_dir():
                frames = sorted(clip_dir.glob("frame_*.jpg"))
                if len(frames) > 0:
                    clip_name = clip_dir.name

                    # Cerca label nelle annotazioni
                    label_info = self._get_label_from_annotations(clip_name)

                    if label_info is not None:
                        self.samples.append({
                            'clip_dir': clip_dir,
                            'clip_name': clip_name,
                            'num_frames': len(frames),
                            'label': label_info['label'],
                            'gender': label_info.get('gender', 'Unknown'),
                            'behavioral_features': label_info.get('features', None)
                        })

        if len(self.samples) == 0:
            raise ValueError(f"Nessuna clip trovata in {root_dir} con annotazioni valide!")

        print(f"Dataset caricato: {len(self.samples)} clips")
        frame_counts = [s['num_frames'] for s in self.samples]
        print(f"Frame per clip - min: {min(frame_counts)}, max: {max(frame_counts)}, "
              f"media: {np.mean(frame_counts):.1f}")

        # Statistiche label
        truth_count = sum(1 for s in self.samples if s['label'] == 0)
        lie_count = sum(1 for s in self.samples if s['label'] == 1)
        print(f"Label distribution - Truth: {truth_count}, Deception: {lie_count}")

    def _load_annotations(self, annotation_file):
        """
        Carica il CSV delle annotazioni MUMIN.
        """
        ann_path = Path(annotation_file)
        if not ann_path.exists():
            raise FileNotFoundError(f"File annotazioni non trovato: {annotation_file}")

        # Leggi CSV (skippa le prime 2 righe che sono header multi-livello)
        df = pd.read_csv(ann_path, skiprows=2, sep=';')

        # Rinomina colonne per facilità (la prima colonna è il filename)
        df.columns = df.columns.str.strip()

        # Crea dizionario: clip_name -> info
        annotations = {}
        for _, row in df.iterrows():
            clip_name = row['File name of the video clip'].strip()
            label_str = row['Label "truth" or "deception"'].strip()

            # Converti label in numero
            label = 0 if label_str.lower() == 'truth' else 1

            # Estrai gender
            gender = row['Participants gender'].strip() if 'Participants gender' in row else 'Unknown'

            # Se richiesto, estrai feature comportamentali (tutti gli altri campi)
            features = None
            if self.use_behavioral_features:
                # Prendi tutte le colonne numeriche (escludendo le prime 3)
                feature_cols = df.columns[3:]  # Skippa filename, label, gender
                features = row[feature_cols].values.astype(float)

            annotations[clip_name] = {
                'label': label,
                'gender': gender,
                'features': features
            }

        print(f"✓ Caricate {len(annotations)} annotazioni da {ann_path.name}")
        return annotations

    def _get_label_from_annotations(self, clip_name):
        """
        Cerca la label per una clip specifica nelle annotazioni.
        """
        # Prova match esatto
        if clip_name in self.annotations:
            return self.annotations[clip_name]

        # Prova match parziale (es. "AN_WILTY_EP15_truth1" vs path che contiene questo)
        for ann_name, info in self.annotations.items():
            if ann_name in clip_name or clip_name in ann_name:
                return info

        # Non trovato
        return None

    def get_default_transform(self):
        """Trasformazioni standard per ResNet pre-trained."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_dir = sample['clip_dir']
        label = sample['label']

        # Carica tutti i frame della clip
        frame_files = sorted(clip_dir.glob("frame_*.jpg"))

        # Opzionale: tronca se troppo lungo
        if self.max_frames is not None and len(frame_files) > self.max_frames:
            frame_files = frame_files[:self.max_frames]

        frames = []
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"⚠ Warning: impossibile leggere {frame_path}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        # Stack frames: shape = (num_frames, C, H, W)
        if len(frames) == 0:
            raise ValueError(f"Nessun frame valido trovato per {clip_dir}")

        frames_tensor = torch.stack(frames)

        # Ritorna anche metadata se servono
        return {
            'frames': frames_tensor,
            'label': label,
            'length': len(frames),
            'clip_name': sample['clip_name'],
            'gender': sample['gender']
        }


def collate_fn_with_padding(batch):
    """
    Collate function che gestisce sequenze di lunghezza variabile con padding.

    Args:
        batch: Lista di dict da __getitem__

    Returns:
        frames: (batch_size, max_seq_len, C, H, W) - con padding
        labels: (batch_size,)
        lengths: (batch_size,) - lunghezze originali prima del padding
        mask: (batch_size, max_seq_len) - True per frame reali, False per padding
        metadata: dict con clip_name, gender per ogni sample
    """
    # Estrai componenti
    frames_list = [item['frames'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]
    clip_names = [item['clip_name'] for item in batch]
    genders = [item['gender'] for item in batch]

    # Trova lunghezza massima nel batch
    max_len = max(lengths)
    batch_size = len(frames_list)
    C, H, W = frames_list[0].shape[1:]

    # Crea tensori con padding
    padded_frames = torch.zeros(batch_size, max_len, C, H, W)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (frames, length) in enumerate(zip(frames_list, lengths)):
        padded_frames[i, :length] = frames
        mask[i, :length] = True  # True = frame reale, False = padding

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    metadata = {
        'clip_names': clip_names,
        'genders': genders
    }

    return padded_frames, labels, lengths, mask, metadata