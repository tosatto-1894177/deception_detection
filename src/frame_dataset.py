import torch
import cv2
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import numpy as np


class DOLOSFrameDataset(Dataset):
    """
    Dataset per DOLOS con annotazioni MUMIN.
    Supporta sia random split che subject-independent split con fold DOLOS.
    """

    def __init__(self, root_dir, annotation_file, transform=None, max_frames=None,
                 use_behavioral_features=False, clip_filter=None):
        """
        Args:
            root_dir: Directory con i frame estratti (organizzati per clip)
            annotation_file: Path al CSV/Excel con annotazioni MUMIN
            transform: Trasformazioni da applicare ai frame
            max_frames: Se specificato, tronca sequenze più lunghe
            use_behavioral_features: Se True, carica feature comportamentali
            clip_filter: Set di clip_name da includere (per subject-independent split)
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

                    # Se clip_filter specificato, filtra
                    if clip_filter is not None and clip_name not in clip_filter:
                        continue

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
        """Carica annotazioni MUMIN da CSV o Excel."""
        ann_path = Path(annotation_file)
        if not ann_path.exists():
            raise FileNotFoundError(f"File annotazioni non trovato: {annotation_file}")

        # Leggi Excel (skippa le prime 2 righe)
        df = pd.read_excel(ann_path, skiprows=2, engine='openpyxl')

        # Pulisci nomi colonne
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

            # Feature comportamentali (se richieste)
            features = None
            if self.use_behavioral_features:
                feature_cols = df.columns[3:]
                features = row[feature_cols].values.astype(float)

            annotations[clip_name] = {
                'label': label,
                'gender': gender,
                'features': features
            }

        print(f"✓ Caricate {len(annotations)} annotazioni da {ann_path.name}")
        return annotations

    def _get_label_from_annotations(self, clip_name):
        """Cerca la label per una clip specifica."""
        # Match esatto
        if clip_name in self.annotations:
            return self.annotations[clip_name]

        # Match parziale
        for ann_name, info in self.annotations.items():
            if ann_name in clip_name or clip_name in ann_name:
                return info

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

        # Tronca se troppo lungo
        if self.max_frames is not None and len(frame_files) > self.max_frames:
            frame_files = frame_files[:self.max_frames]

        frames = []
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"⚠️  Warning: impossibile leggere {frame_path}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        if len(frames) == 0:
            raise ValueError(f"Nessun frame valido trovato per {clip_dir}")

        frames_tensor = torch.stack(frames)

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
    """
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
        mask[i, :length] = True

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    metadata = {
        'clip_names': clip_names,
        'genders': genders
    }

    return padded_frames, labels, lengths, mask, metadata


# ============ SUBJECT-INDEPENDENT SPLIT UTILITIES ============

def load_dolos_fold(fold_path):
    """
    Carica un fold DOLOS (train o test).

    Args:
        fold_path: Path al CSV del fold (es. 'data/splits/train_fold1.csv')

    Returns:
        list: Lista di clip_name nel fold
    """
    fold_path = Path(fold_path)

    if not fold_path.exists():
        raise FileNotFoundError(f"Fold file non trovato: {fold_path}")

    # Leggi CSV
    df = pd.read_csv(fold_path)

    # Trova colonna con clip names
    if 'clip_name' in df.columns:
        clip_names = df['clip_name'].tolist()
    elif 'File name of the video clip' in df.columns:
        clip_names = df['File name of the video clip'].tolist()
    else:
        # Usa prima colonna
        clip_names = df.iloc[:, 0].tolist()

    # Pulisci nomi
    clip_names = [str(name).strip() for name in clip_names]

    print(f"✓ Fold {fold_path.name}: {len(clip_names)} clip")

    return clip_names


def create_subject_independent_split(frames_dir, annotation_file,
                                     train_fold_path, test_fold_path,
                                     val_ratio=0.2, seed=42):
    """
    Crea split subject-independent usando fold DOLOS + validation split.

    Strategy:
    1. Carica train fold DOLOS (subject-independent)
    2. Carica test fold DOLOS (subject-independent)
    3. Splita train fold → 80% train, 20% validation (mantiene subject-independence)

    Args:
        frames_dir: Directory con frame estratti
        annotation_file: File annotazioni MUMIN
        train_fold_path: Path al train fold DOLOS
        test_fold_path: Path al test fold DOLOS
        val_ratio: % di train da usare per validation
        seed: Random seed

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print("CREATING SUBJECT-INDEPENDENT SPLIT")
    print("=" * 70)

    # 1. Carica fold DOLOS
    train_clips = load_dolos_fold(train_fold_path)
    test_clips = load_dolos_fold(test_fold_path)

    print(f"\nDOLOS fold originali:")
    print(f"  Train fold: {len(train_clips)} clip")
    print(f"  Test fold:  {len(test_clips)} clip")

    # 2. Crea dataset completo per verificare quali clip esistono
    full_dataset = DOLOSFrameDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=None  # Carica tutte le clip disponibili
    )

    available_clips = set(s['clip_name'] for s in full_dataset.samples)

    # 3. Filtra solo clip disponibili
    train_clips_available = [c for c in train_clips if c in available_clips]
    test_clips_available = [c for c in test_clips if c in available_clips]

    print(f"\nClip effettivamente disponibili:")
    print(f"  Train: {len(train_clips_available)}/{len(train_clips)} "
          f"({len(train_clips_available) / len(train_clips) * 100:.1f}%)")
    print(f"  Test:  {len(test_clips_available)}/{len(test_clips)} "
          f"({len(test_clips_available) / len(test_clips) * 100:.1f}%)")

    # 4. Split train → train + validation (subject-independent)
    # Raggruppa per soggetto (estratto da clip name)
    train_by_subject = {}
    for clip in train_clips_available:
        # Estrai subject ID (es. "AN_WILTY_EP15" da "AN_WILTY_EP15_truth1")
        parts = clip.split('_')
        if len(parts) > 1:
            subject_id = '_'.join(parts[:-1])  # Rimuovi ultimo token
        else:
            subject_id = clip  # Fallback

        if subject_id not in train_by_subject:
            train_by_subject[subject_id] = []
        train_by_subject[subject_id].append(clip)

    subjects = list(train_by_subject.keys())
    n_subjects = len(subjects)
    n_val_subjects = max(1, int(n_subjects * val_ratio))

    print(f"\nSubject split:")
    print(f"  Soggetti totali: {n_subjects}")
    print(f"  Soggetti per validation: {n_val_subjects}")
    print(f"  Soggetti per training: {n_subjects - n_val_subjects}")

    # Random shuffle subjects
    np.random.shuffle(subjects)

    val_subjects = subjects[:n_val_subjects]
    train_subjects = subjects[n_val_subjects:]

    # Assegna clip a train/val
    train_final = []
    val_final = []

    for subject in train_subjects:
        train_final.extend(train_by_subject[subject])

    for subject in val_subjects:
        val_final.extend(train_by_subject[subject])

    print(f"\nFinal split:")
    print(f"  Train: {len(train_final)} clip da {len(train_subjects)} soggetti")
    print(f"  Val:   {len(val_final)} clip da {len(val_subjects)} soggetti")
    print(f"  Test:  {len(test_clips_available)} clip (soggetti separati)")

    # 5. Crea dataset con filtri
    train_dataset = DOLOSFrameDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=set(train_final)
    )

    val_dataset = DOLOSFrameDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=set(val_final)
    )

    test_dataset = DOLOSFrameDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file,
        clip_filter=set(test_clips_available)
    )

    # Verifica no overlap
    train_set = set(s['clip_name'] for s in train_dataset.samples)
    val_set = set(s['clip_name'] for s in val_dataset.samples)
    test_set = set(s['clip_name'] for s in test_dataset.samples)

    assert len(train_set & val_set) == 0, "❌ Overlap tra train e val!"
    assert len(train_set & test_set) == 0, "❌ Overlap tra train e test!"
    assert len(val_set & test_set) == 0, "❌ Overlap tra val e test!"

    print(f"\n✅ Subject-independent split creato con successo!")
    print(f"   Nessun overlap tra split verificato.")
    print("=" * 70 + "\n")

    return train_dataset, val_dataset, test_dataset


def create_random_split(frames_dir, annotation_file,
                        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                        seed=42):
    """
    Crea split random (NON subject-independent).
    Utile per test rapidi

    Args:
        frames_dir: Directory con frame estratti
        annotation_file: File annotazioni
        train_ratio: Frazione per training
        val_ratio: Frazione per validation
        test_ratio: Frazione per test
        seed: Random seed

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Le ratio devono sommare a 1.0"

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "=" * 70)
    print("⚠️  CREATING RANDOM SPLIT (NOT SUBJECT-INDEPENDENT)")
    print("=" * 70)

    # Crea dataset completo
    full_dataset = DOLOSFrameDataset(
        root_dir=frames_dir,
        annotation_file=annotation_file
    )

    n_samples = len(full_dataset)
    indices = np.random.permutation(n_samples)

    # Calcola split points
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Crea subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"\nRandom split creato:")
    print(f"  Train: {len(train_dataset)} clip ({train_ratio * 100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} clip ({val_ratio * 100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} clip ({test_ratio * 100:.0f}%)")
    print("=" * 70 + "\n")

    return train_dataset, val_dataset, test_dataset


def create_k_fold_cross_validation(frames_dir, annotation_file, k=3):
    """
    Crea K-fold cross-validation usando i fold DOLOS.

    Args:
        frames_dir: Directory frame
        annotation_file: File annotazioni
        k: Numero di fold (deve matchare i fold DOLOS disponibili)

    Returns:
        list: Lista di tuple (train_dataset, val_dataset, test_dataset) per ogni fold
    """
    print("\n" + "=" * 70)
    print(f"CREATING {k}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    folds_data = []

    for fold_idx in range(1, k + 1):
        print(f"\n--- Fold {fold_idx}/{k} ---")

        # Paths ai fold DOLOS
        train_fold_path = Path(f'data/splits/train_fold{fold_idx}.csv')
        test_fold_path = Path(f'data/splits/test_fold{fold_idx}.csv')

        # Crea split per questo fold
        train_ds, val_ds, test_ds = create_subject_independent_split(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            train_fold_path=train_fold_path,
            test_fold_path=test_fold_path,
            val_ratio=0.2,
            seed=42 + fold_idx  # Seed diverso per ogni fold
        )

        folds_data.append((train_ds, val_ds, test_ds))

    print(f"\n✅ {k}-fold cross-validation setup completato!")
    print(f"   Esegui training su ogni fold e calcola metriche medie.")
    print("=" * 70 + "\n")

    return folds_data


# ============ ESEMPIO D'USO ============

if __name__ == "__main__":
    """
    Test del sistema di split.
    """
    print("TEST SUBJECT-INDEPENDENT SPLIT")
    print("=" * 70)

    frames_dir = "data/frames"
    annotation_file = "data/metadata/train_dolos.xlsx"

    # Opzione 1: Subject-independent con fold DOLOS
    print("\n[OPZIONE 1] Subject-Independent Split (RACCOMANDATO)")
    try:
        train_ds, val_ds, test_ds = create_subject_independent_split(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            train_fold_path="data/splits/train_fold1.csv",
            test_fold_path="data/splits/test_fold1.csv",
            val_ratio=0.2
        )
        print(f"✅ Split creato con successo!")
    except FileNotFoundError as e:
        print(f"⚠️  File fold non trovato: {e}")
        print(f"   Verifica che i file train_fold*.csv esistano in data/splits/")

    # Opzione 2: Random split (per test rapidi)
    print("\n[OPZIONE 2] Random Split (SOLO PER TEST)")
    train_ds, val_ds, test_ds = create_random_split(
        frames_dir=frames_dir,
        annotation_file=annotation_file,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Opzione 3: K-fold cross-validation
    print("\n[OPZIONE 3] 3-Fold Cross-Validation")
    try:
        folds = create_k_fold_cross_validation(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            k=3
        )
        print(f"✅ {len(folds)} fold creati!")
    except FileNotFoundError as e:
        print(f"⚠️  File fold non trovati: {e}")

    print("\n" + "=" * 70)
    print("✅ Test completato!")
    print("=" * 70)