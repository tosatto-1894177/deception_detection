"""
Creazione Dataset con split DOLOS
"""

import torch
import cv2
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from openface_loader import OpenFaceLoader


class DOLOSDataset(Dataset):
    """
    Dataset DOLOS con annotazioni MUMIN.
    Supporta 2 tipi di split e un numero variabile di frame:
    - Sampling dinamico nel __getitem__
    - Gestisce clip da 10 a 6000+ frames
    - Sampling intelligente per mantenere informazione temporale
    """

    def __init__(self, root_dir, annotation_file, transform=None, max_frames=50,
                 clip_filter=None, use_openface=False, openface_csv_dir=None, verbose=True):
        """
        Args:
            root_dir: Cartella dove sono contenuti i frame estratti dalle clip
            annotation_file: percorso dell'Excel che contiene le annotazioni MUMIN
            transform: funzione transform
            max_frames: Numero massimo di frame da caricare per ciascuna clip
            clip_filter: Insieme delle clip da includere (per splits)
            use_openface: se True carica anche le feature Openface
            openface_csv_dir: Cartella dove sono contenuti i csv con le feature Openface
            verbose: per gestire i print
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or self.get_default_transform()
        self.max_frames = max_frames
        self.use_openface = use_openface

        self.openface_loader = None
        if use_openface:
            if openface_csv_dir is None:
                raise ValueError("openface_csv_dir deve essere specificato se use_openface=True")

            self.openface_loader = OpenFaceLoader(
                csv_dir=openface_csv_dir,
                use_aus=True,
                use_gaze=True,
                use_pose=True,
                normalize=True
            )

        # Carica annotazioni Mumin
        self.annotations = self._load_annotations(annotation_file, verbose=verbose)

        # Trova clip con frame
        self.samples = []
        for clip_dir in sorted(self.root_dir.rglob("*")):
            if clip_dir.is_dir():
                frames = sorted(clip_dir.glob("frame_*.jpg"))
                if len(frames) > 0:
                    clip_name = clip_dir.name

                    # Filtra se necessario
                    if clip_filter is not None and clip_name not in clip_filter:
                        continue

                    # Cerca label
                    label_info = self._get_label_from_annotations(clip_name)

                    if label_info is not None:
                        self.samples.append({
                            'clip_dir': clip_dir,
                            'clip_name': clip_name,
                            'num_frames': len(frames),
                            'label': label_info['label'],
                            'gender': label_info.get('gender', 'Unknown')
                        })

        if len(self.samples) == 0:
            raise ValueError(f"Nessuna clip trovata in {root_dir}")

        if verbose:
            print(f"Dataset caricato: {len(self.samples)} clips")
            frame_counts = [s['num_frames'] for s in self.samples]
            print(f"Frame per clip - min: {min(frame_counts)}, max: {max(frame_counts)}, "
                  f"media: {np.mean(frame_counts):.1f}")

            # Statistiche label
            truth_count = sum(1 for s in self.samples if s['label'] == 0)
            lie_count = sum(1 for s in self.samples if s['label'] == 1)
            print(f"Label distribution - Truth: {truth_count}, Deception: {lie_count}")
            print(f"\n")

    def _load_annotations(self, annotation_file, verbose=True):
        """Carica informazioni da Excel"""
        ann_path = Path(annotation_file)
        if not ann_path.exists():
            raise FileNotFoundError(f"File con annotazioni non trovato: {annotation_file}")

        # Leggi Excel
        df = pd.read_excel(ann_path, skiprows=2, engine='openpyxl')
        df.columns = df.columns.str.strip()

        annotations = {}
        for _, row in df.iterrows():
            clip_name = row['File name of the video clip'].strip()
            label_str = row['Label "truth" or "deception"'].strip()
            label = 0 if label_str.lower() == 'truth' else 1
            gender = row['Participants gender'].strip() if 'Participants gender' in row else 'Unknown'

            annotations[clip_name] = {
                'label': label,
                'gender': gender
            }

        if verbose:
            print(f"✅ Caricate {len(annotations)} annotazioni")
        return annotations

    def _get_label_from_annotations(self, clip_name):
        """Cerca label per clip"""
        if clip_name in self.annotations:
            return self.annotations[clip_name]

        return None

    def _sample_frames(self, frame_files, max_frames):
        """
        Campiona frame da lista completa.

        Args:
            frame_files: Lista Path di tutti i frame
            max_frames: Numero massimo frame da caricare

        Returns:
            sampled_files: Lista Path frame campionati (len <= max_frames)
        """
        total_frames = len(frame_files)

        if total_frames <= max_frames:
            # Clip corta → usa tutti i frame
            return frame_files

        # Clip lunga → campiona
        # Sampling uniforme: distribuisci frame equamente su tutta la clip
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        return [frame_files[i] for i in indices]

    def get_default_transform(self):
        """Trasformazioni standard per ResNet."""
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

        # Carica tutti i frame disponibili
        frame_files = sorted(clip_dir.glob("frame_*.jpg"))

        # Sampling dinamico se troppi frame
        sampled_files = self._sample_frames(
            frame_files,
            self.max_frames
        )

        # Carica frame campionati
        frames = []
        for frame_path in sampled_files:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"  Warning: impossibile leggere {frame_path}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        if len(frames) == 0:
            raise ValueError(f"Nessun frame valido per {clip_dir}")

        frames_tensor = torch.stack(frames)

        openface_features = None
        if self.use_openface and self.openface_loader is not None:
            clip_name = sample['clip_name']
            openface_features = self.openface_loader.load_and_preprocess(clip_name)

            # Gestione frame mismatch
            if openface_features is not None:
                openface_features = self._align_openface_to_frames(
                    openface_features,
                    len(frames)
                )

        return {
            'frames': frames_tensor,
            'label': label,
            'length': len(frames),
            'clip_name': sample['clip_name'],
            'gender': sample['gender'],
            'openface_features': openface_features
        }

    def _align_openface_to_frames(self, openface_features, num_frames):
        """
        Allinea numero frame OpenFace con numero frame video

        Args:
            openface_features: (T_of, feature_dim) tensor
            num_frames: numero frame video caricati

        Returns:
            (num_frames, feature_dim) tensor allineato
        """
        T_of = openface_features.shape[0]

        if T_of == num_frames:
            # Già allineato
            return openface_features

        elif T_of < num_frames:
            # OpenFace ha meno frame → padding con zeros
            pad_size = num_frames - T_of
            padding = torch.zeros(pad_size, openface_features.shape[1])
            return torch.cat([openface_features, padding], dim=0)

        else:
            # OpenFace ha più frame → sampling uniforme
            indices = torch.linspace(0, T_of - 1, num_frames).long()
            return openface_features[indices]

def collate_fn_with_padding(batch):
    """
    Collate function con padding per sequenze variabili

    Args:
        batch: lista di dict, uno per ogni clip:
        [{'frames': tensor(n, 3, 224, 224), 'label': 0/1, 'length': n, 'clip_name': 'NOME_EPX_lie/truth', 'gender': 'gender', 'openface': feature_list[]},
            ...]
            dove n = numero di frame della clip
    """

    # Da lista di dict a liste separate per ciascun componente
    frames_list = [item['frames'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]
    clip_names = [item['clip_name'] for item in batch]
    genders = [item['gender'] for item in batch]
    openface_list = [item.get('openface_features', None) for item in batch]

    # Trova la lunghezza massima -> questa diventa la dimensione temporale del batch padded
    # Tutte le clip più corte verranno "allungate" a max_len
    max_len = max(lengths)
    batch_size = len(frames_list)
    C, H, W = frames_list[0].shape[1:]

    # Alloca tensor pieno di ZERI
    padded_frames = torch.zeros(batch_size, max_len, C, H, W)
    # Alloca mask pieno di False
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Alloca tensor per OpenFace features se presente
    openface_features_padded = None
    if openface_list[0] is not None:
        feature_dim = openface_list[0].shape[1]
        openface_features_padded = torch.zeros(batch_size, max_len, feature_dim)

    # Per ogni clip nel batch
    for i, (frames, length) in enumerate(zip(frames_list, lengths)):
        # Copia i frame della clip, il resto lascia zeri
        padded_frames[i, :length] = frames
        # Segna nella mask quali posizioni hanno dati reali (se False -> padding)
        mask[i, :length] = True

        # Copia le features OpenFace se presenti
        if openface_list[i] is not None:
            openface_features_padded[i, :length] = openface_list[i]

    # Converte labels e lenghts in tensor
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    metadata = {
        'clip_names': clip_names,
        'genders': genders
    }

    return padded_frames, labels, lengths, mask, metadata, openface_features_padded


def load_dolos_fold(fold_path):
    """Carica il CSV con fold DOLOS"""
    fold_path = Path(fold_path)
    if not fold_path.exists():
        raise FileNotFoundError(f"File CSV non trovato: {fold_path}")

    # Legge il CSV composto da una riga così impostata: nome_clip, truth/deception, gender
    df = pd.read_csv(fold_path, header=None, names=['clip_name', 'label', 'gender'])

    clip_names = df['clip_name'].str.strip().tolist()

    return clip_names

def create_dolos_fold_split(frames_dir, annotation_file,
                            train_fold_path, test_fold_path,
                            val_ratio=0.2, seed=42, max_frames=50,
                            use_openface=False, openface_csv_dir=None):
    """
    Usa i fold DOLOS ufficiali

    Split del train fold tra train e val randomicamente

    Args:
        frames_dir: Cartella dove sono contenuti i frame estratti dalle clip
        annotation_file: percorso dell'Excel che contiene le annotazioni MUMIN
        train_fold_path: percorso al CSV che contiene il fold DOLOS per il train
        test_fold_path: percorso al CSV che contiene il fold DOLOS per il test
        val_ratio: % delle clip del fold train che verranno usate per validation
        seed: Random seed
        max_frames: Numero max di frame supportato per ciascuna clip
        use_openface: se True carica anche le feature Openface
        openface_csv_dir: Cartella dove sono contenuti i csv con le feature Openface

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print("📁 Utilizzo i fold ufficiali DOLOS")
    print("=" * 70)

    # Carica fold DOLOS
    train_clips = load_dolos_fold(train_fold_path)
    test_clips = load_dolos_fold(test_fold_path)

    print(f"  Train fold: {len(train_clips)} clip")
    print(f"  Test fold:  {len(test_clips)} clip")

    # Crea dataset completo
    dataset = DOLOSDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=None,
        max_frames=max_frames,
        use_openface=use_openface, 
        openface_csv_dir=openface_csv_dir,
        verbose=False
    )

    available_clips = set(s['clip_name'] for s in dataset.samples)

    # Filtra clip disponibili
    train_clips_available = [c for c in train_clips if c in available_clips]
    test_clips_available = [c for c in test_clips if c in available_clips]

    print(f"\nClip disponibili:")
    print(f"  Train: {len(train_clips_available)}/{len(train_clips)}")
    print(f"  Test:  {len(test_clips_available)}/{len(test_clips)}")

    # Split train → train/val
    n_train = len(train_clips_available)
    n_val = int(n_train * val_ratio)

    # Shuffle e split
    shuffled = train_clips_available.copy()
    np.random.shuffle(shuffled)

    train_final = shuffled[n_val:]
    val_final = shuffled[:n_val]

    print(f"\nClip presenti in ciascun dataset:")
    print(f"  Train: {len(train_final)} clip")
    print(f"  Val:   {len(val_final)} clip")
    print(f"  Test:  {len(test_clips_available)} clip")

    # Crea dataset con filtri
    train_dataset = DOLOSDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=set(train_final),
        max_frames=max_frames,
        use_openface=use_openface,
        openface_csv_dir=openface_csv_dir,
        verbose=True
    )

    val_dataset = DOLOSDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=set(val_final),
        max_frames=max_frames,
        use_openface=use_openface,
        openface_csv_dir=openface_csv_dir,
        verbose=True
    )

    test_dataset = DOLOSDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=set(test_clips_available),
        max_frames=max_frames,
        use_openface=use_openface,
        openface_csv_dir=openface_csv_dir,
        verbose=True
    )

    print(f"\n✅ Creati Dataset con split DOLOS!")
    print("=" * 80 + "\n")

    return train_dataset, val_dataset, test_dataset
