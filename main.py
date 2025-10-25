"""
Deception Detection - Main Pipeline
Gestisce preprocessing, training e testing del modello.
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from config_loader import load_config
from scripts.frame_extractor import process_dolos_with_faces
from src.frame_dataset import DOLOSFrameDataset, collate_fn_with_padding


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deception Detection Pipeline')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path al file di configurazione')

    parser.add_argument('--mode', type=str,
                        choices=['preprocess', 'train', 'test'],
                        required=True,
                        help='Modalità di esecuzione')

    # Override parametri training
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=int, help='Override num epochs')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='Override device')

    return parser.parse_args()


def preprocess(config):
    """
    Estrai frame dai video con face detection.
    Eseguito UNA VOLTA SOLA all'inizio.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING: Estrazione frame con face detection")
    print("=" * 70)

    video_dir = Path(config.get('paths.video_dir'))
    frames_dir = Path(config.get('paths.frames_dir'))

    if not video_dir.exists():
        raise FileNotFoundError(f"Directory video non trovata: {video_dir}")

    frames_dir.mkdir(parents=True, exist_ok=True)

    process_dolos_with_faces(
        video_dir=str(video_dir),
        output_base_dir=str(frames_dir),
        fps=config.get('preprocessing.fps'),
        img_size=tuple(config.get('preprocessing.img_size')),
        device=config.get('preprocessing.face_detection.device')
    )

    print(f"\n✓ Preprocessing completato!")
    print(f"  Frame salvati in: {frames_dir}")


def create_dataloaders(config):
    """Crea DataLoader per train/val/test."""
    print("\n" + "=" * 70)
    print("CARICAMENTO DATASET")
    print("=" * 70)

    frames_dir = config.get('paths.frames_dir')
    batch_size = config['training']['batch_size']

    # Training set
    train_dataset = DOLOSFrameDataset(
        root_dir=frames_dir,
        annotation_file=config.get('paths.train_annotations'),
        max_frames=config.get('preprocessing.max_frames'),
        use_behavioral_features=config.get('dataset.use_behavioral_features')
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    # Validation set (opzionale)
    val_loader = None
    val_ann = config.get('paths.val_annotations')
    if val_ann and Path(val_ann).exists():
        val_dataset = DOLOSFrameDataset(
            root_dir=frames_dir,
            annotation_file=val_ann,
            max_frames=config.get('preprocessing.max_frames'),
            use_behavioral_features=config.get('dataset.use_behavioral_features')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_padding,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )

    print(f"\n✓ Dataset caricati:")
    print(f"  Train: {len(train_dataset)} clips")
    if val_loader:
        print(f"  Validation: {len(val_dataset)} clips")

    return train_loader, val_loader


def build_model(config):
    """
    Costruisci modello (ResNet feature extractor per ora).
    TODO: Aggiungere TCN + Attention + MLP dopo.
    """
    print("\n" + "=" * 70)
    print("COSTRUZIONE MODELLO")
    print("=" * 70)

    from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights

    resnet_arch = config.get('model.resnet.architecture')
    pretrained = config.get('model.resnet.pretrained')

    if resnet_arch == 'resnet34':
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)
        feature_dim = 512
    else:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        feature_dim = 2048

    # Rimuovi layer finale
    resnet.fc = torch.nn.Identity()

    # Freeze se richiesto
    if config.get('model.resnet.freeze_layers'):
        for param in resnet.parameters():
            param.requires_grad = False

    device = torch.device(config['training']['device'])
    resnet = resnet.to(device)

    trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in resnet.parameters())

    print(f"\n✓ Modello: {resnet_arch}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Parametri totali: {total:,}")
    print(f"  Parametri trainable: {trainable:,}")
    print(f"  Device: {device}")

    return resnet


def train(config, model, train_loader, val_loader=None):
    """
    Training loop.
    TODO: Implementare loop completo dopo aver creato TCN+Attention.
    Per ora solo test forward pass.
    """
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    device = torch.device(config['training']['device'])
    num_epochs = config['training']['num_epochs']

    print(f"\nConfigurazione training:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Optimizer: {config['training']['optimizer']}")

    # Test forward pass
    print("\nTest forward pass...")
    model.eval()
    with torch.no_grad():
        frames, labels, lengths, mask, metadata = next(iter(train_loader))
        frames = frames.to(device)

        batch_size, max_len, C, H, W = frames.shape
        frames_flat = frames.view(batch_size * max_len, C, H, W)

        features = model(frames_flat)
        features = features.view(batch_size, max_len, -1)

        print(f"\n✓ Forward pass OK!")
        print(f"  Input: {frames.shape}")
        print(f"  Output: {features.shape}")
        print(f"  Labels: {labels}")

    print("\n⚠ Training loop completo da implementare dopo TCN+Attention")

    return model


def test(config, model):
    """
    Test finale.
    TODO: Implementare dopo training.
    """
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)
    print("⚠ Da implementare")


def main():
    """Main entry point."""
    args = parse_args()

    # Carica config
    config = load_config(args.config)

    # Override da CLI
    if args.batch_size:
        config.update({'training': {'batch_size': args.batch_size}})
    if args.lr:
        config.update({'training': {'learning_rate': args.lr}})
    if args.epochs:
        config.update({'training': {'num_epochs': args.epochs}})
    if args.device:
        config.update({'training': {'device': args.device}})

    # Crea directory
    config.create_directories()

    # Esegui modalità richiesta
    if args.mode == 'preprocess':
        preprocess(config)

    elif args.mode == 'train':
        train_loader, val_loader = create_dataloaders(config)
        model = build_model(config)
        model = train(config, model, train_loader, val_loader)

        # Salva checkpoint
        checkpoint_dir = Path(config.get('paths.models_dir'))
        checkpoint_path = checkpoint_dir / 'resnet_features.pth'