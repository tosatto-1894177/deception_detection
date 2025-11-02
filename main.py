"""
Deception Detection - Main Pipeline
Gestisce preprocessing, training e testing del modello.
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.config_loader import load_config
from scripts.frame_extractor import process_dolos_with_faces
from src.frame_dataset import DOLOSFrameDataset, collate_fn_with_padding
from src.train import train_model

# Ottimizzazioni memoria GPU
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deception Detection Pipeline')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path al file di configurazione')

    parser.add_argument('--mode', type=str,
                        choices=['preprocess', 'train', 'test', 'demo'],
                        required=True,
                        help='Modalità di esecuzione')

    # Override parametri training
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=int, help='Override num epochs')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='Override device')

    # Split type
    parser.add_argument('--subject_split', action='store_true',
                        help='Usa subject-independent split (RACCOMANDATO per training finale)')

    return parser.parse_args()


def preprocess(config):
    """
    Estrai frame dai video con face detection.
    Eseguito UNA VOLTA SOLA all'inizio.
    """
    print("\n" + "="*70)
    print("PREPROCESSING: Estrazione frame con face detection")
    print("="*70)

    video_dir = Path(config.get('paths.video_dir'))
    frames_dir = Path(config.get('paths.frames_dir'))

    print(f"  Cartella delle clip: {video_dir.resolve()}")
    print(f"  Frame da salvare in: {frames_dir.resolve()}")

    if not video_dir.exists():
        raise FileNotFoundError(f"Directory video non trovata: {video_dir}")

    frames_dir.mkdir(parents=True, exist_ok=True)

    process_dolos_with_faces(
        video_dir=str(video_dir),
        output_base_dir=str(frames_dir),
        fps=config.get('preprocessing.fps'),
        img_size=tuple(config.get('preprocessing.img_size')),
        device=config.get('preprocessing.face_detection.device'),
        save_failed_frames=config.get('preprocessing.face_detection.save_failed_frames')
    )

    print(f"\n✓ Preprocessing completato!")
    print(f"  Frame salvati in: {frames_dir}")


def create_dataloaders(config, use_subject_split=False):
    """
    Crea DataLoader per train/val/test.

    Args:
        config: Config object
        use_subject_split: Se True, usa subject-independent split con fold DOLOS
                          Se False, usa random split (più veloce per test)
    """
    print("\n" + "="*70)
    print("CARICAMENTO DATASET")
    print("="*70)

    frames_dir = config.get('paths.frames_dir')
    batch_size = config['training']['batch_size']
    annotation_file = config.get('paths.train_annotations')

    if use_subject_split:
        # Subject-Independent Split (RACCOMANDATO per training finale)
        print("\n🎓 Usando SUBJECT-INDEPENDENT SPLIT")

        from src.frame_dataset import create_subject_independent_split

        # Path ai fold DOLOS (puoi cambiare fold_idx nel config)
        fold_idx = config.get('dataset.fold_idx', 1)
        train_fold_path = Path(f'data/splits/train_fold{fold_idx}.csv')
        test_fold_path = Path(f'data/splits/test_fold{fold_idx}.csv')

        train_dataset, val_dataset, test_dataset = create_subject_independent_split(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            train_fold_path=train_fold_path,
            test_fold_path=test_fold_path,
            val_ratio=config.get('dataset.val_ratio', 0.2),
            seed=config.get('dataset.seed', 42)
        )

    else:
        # Random Split (VELOCE per test/demo)
        print("\n⚡ Usando RANDOM SPLIT (veloce per test)")

        from src.frame_dataset import create_random_split

        train_dataset, val_dataset, test_dataset = create_random_split(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            train_ratio=config.get('dataset.train_ratio', 0.7),
            val_ratio=config.get('dataset.val_ratio', 0.15),
            test_ratio=config.get('dataset.test_ratio', 0.15),
            seed=config.get('dataset.seed', 42)
        )

    # Crea DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_padding,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_padding,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    print(f"\n✓ Dataset caricati:")
    print(f"  Train: {len(train_dataset)} clips")
    print(f"  Val:   {len(val_dataset)} clips")
    print(f"  Test:  {len(test_dataset)} clips")

    return train_loader, val_loader, test_loader


def test_with_real_batch(config, use_subject_split=False):
    """
    Test con un batch reale dal dataset.
    Verifica che tutto funzioni prima del training completo.
    """
    print("\n" + "="*70)
    print("TEST CON BATCH REALE")
    print("="*70)

    # Crea dataloader
    train_loader, _, _ = create_dataloaders(config, use_subject_split)

    # Build model
    from src.model import build_model
    model, device = build_model(config)

    print("\nCaricamento primo batch...")
    frames, labels, lengths, mask, metadata = next(iter(train_loader))

    print(f"\nBatch info:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Labels: {labels}")
    print(f"  Lengths: {lengths}")
    print(f"  Clip names: {metadata['clip_names']}")

    # Forward pass
    print("\nForward pass...")
    frames = frames.to(device)
    mask = mask.to(device)

    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(frames, mask)

    print(f"\nOutput:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")

    # Predictions
    predictions = torch.argmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)

    print(f"\nPredictions:")
    for i in range(len(predictions)):
        pred_class = 'Truth' if predictions[i] == 0 else 'Deception'
        true_class = 'Truth' if labels[i] == 0 else 'Deception'
        conf = probabilities[i, predictions[i]].item()

        print(f"  {metadata['clip_names'][i]}:")
        print(f"    True: {true_class}, Predicted: {pred_class}, Confidence: {conf:.3f}")

    print(f"\n✓ Test con batch reale completato!")
    print(f"  Il modello è pronto per training completo")


def demo_mode(config):
    """
    Modalità demo: mini-training su subset piccolo per test rapido.
    Utile per verificare che tutto funzioni prima del training completo.
    """
    print("\n" + "="*70)
    print("DEMO MODE: Mini-training su subset ridotto")
    print("="*70)

    # Override config per demo veloce
    original_epochs = config['training']['num_epochs']
    config.update({
        'training': {
            'num_epochs': 3,
            'batch_size': 2
        }
    })

    print(f"\nConfigurazione demo:")
    print(f"  Epochs: 3 (original: {original_epochs})")
    print(f"  Batch size: 2")
    print(f"  Device: {config['training']['device']}")
    print(f"  Split: Random (veloce)")

    # Crea dataloaders con RANDOM split (più veloce per demo)
    train_loader, val_loader, test_loader = create_dataloaders(config, use_subject_split=False)

    # Limita a pochi batch per demo
    print(f"\n⚠️  Demo mode: usando solo 20 samples train, 6 val")

    from torch.utils.data import Subset
    import numpy as np

    # Prendi solo primi N samples
    train_indices = np.arange(min(20, len(train_loader.dataset)))
    val_indices = np.arange(min(6, len(val_loader.dataset)))

    train_subset = Subset(train_loader.dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn_with_padding
    )

    val_subset = Subset(val_loader.dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn_with_padding
    )

    # Training
    trainer = train_model(config, train_loader, val_loader)

    print(f"\n✓ Demo completato!")
    print(f"  Se tutto è OK, esegui training completo con --mode train")


def train(config, use_subject_split=False):
    """
    Training completo del modello.

    Args:
        config: Config object
        use_subject_split: Se True, usa subject-independent split
    """
    print("\n" + "="*70)
    print("TRAINING COMPLETO")
    print("="*70)

    # Crea dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, use_subject_split)

    # Training
    trainer = train_model(config, train_loader, val_loader)

    # Salva modello finale
    final_model_path = Path(config.get('paths.models_dir')) / 'final_model.pth'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config.to_dict(),
        'best_val_f1': trainer.best_val_f1
    }, final_model_path)

    print(f"\n✓ Modello finale salvato: {final_model_path}")

    return trainer


def test(config):
    """
    Test finale con best model.
    """
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)

    # TODO: Implementare test completo
    # - Carica best_model.pth
    # - Carica test set
    # - Inference
    # - Calcola metriche finali
    # - Genera grafici

    print("⚠️  Test mode da implementare")
    print("   Per ora usa validation metrics dal training")


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

    elif args.mode == 'demo':
        demo_mode(config)

    elif args.mode == 'train':
        # Determina se usare subject split
        use_subject_split = args.subject_split

        if use_subject_split:
            print("\n🎓 Training con SUBJECT-INDEPENDENT SPLIT")
            print("   (Corretto per paper/tesi)")
        else:
            print("\n⚡ Training con RANDOM SPLIT")
            print("   (Veloce per test, ma NON per paper/tesi)")

        # Prima testa con batch reale
        test_with_real_batch(config, use_subject_split)

        # Poi training completo
        train(config, use_subject_split)

    elif args.mode == 'test':
        test(config)

    print("\n" + "="*70)
    print("✓ COMPLETATO")
    print("="*70)


if __name__ == "__main__":
    main()