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
from src.frame_dataset import DOLOSFrameDataset, collate_fn_with_padding, load_dolos_fold
from src.train import train_model
from datetime import datetime
from src.model import build_model
from torch.utils.data import Subset
import numpy as np
from src.metrics import MetricsTracker, MetricsVisualizer
from tqdm import tqdm
from src.frame_dataset import create_random_split
import json

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
                        help='Usa subject-independent split custom (per training finale)')

    # DOLOS type
    parser.add_argument('--dolos_fold', action='store_true',
                        help='Usa fold DOLOS ufficiali (comparabile letteratura)')

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


def create_dataloaders(config, use_subject_split=False, use_dolos_fold=False):
    """
    Crea DataLoader per train/val/test.

    Args:
        config: Config object
        use_subject_split: Se True, usa subject-independent split custom
        use_dolos_fold: Se True, usa fold DOLOS ufficiali (non subject-independent)
        Se entrambi False, usa random split (più veloce per test)
    """
    print("\n" + "="*70)
    print("CARICAMENTO DATASET")
    print("="*70)

    frames_dir = config.get('paths.frames_dir')
    batch_size = config['training']['batch_size']
    annotation_file = config.get('paths.train_annotations')

    if use_dolos_fold:
        # ✅ NUOVO: Usa fold DOLOS ufficiali (comparabile con letteratura)
        print("\n📁 Usando FOLD DOLOS UFFICIALI")
        print("   ⚠️  NON subject-independent (overlap soggetti)")

        from src.frame_dataset import create_dolos_fold_split

        fold_idx = config.get('dataset.fold_idx', 1)
        train_fold_path = Path(f'data/splits/train_fold{fold_idx}.csv')
        test_fold_path = Path(f'data/splits/test_fold{fold_idx}.csv')

        train_dataset, val_dataset, test_dataset = create_dolos_fold_split(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            train_fold_path=train_fold_path,
            test_fold_path=test_fold_path,
            val_ratio=config.get('dataset.val_ratio', 0.2),
            seed=config.get('dataset.seed', 42),
            max_frames=50
        )

    elif use_subject_split:
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


def test_with_real_batch(config, use_subject_split=False, use_dolos_fold=False):
    """
    Test con un batch reale dal dataset.
    Verifica che tutto funzioni prima del training completo.
    """
    print("\n" + "="*70)
    print("TEST CON BATCH REALE")
    print("="*70)

    # Crea dataloader
    train_loader, _, _ = create_dataloaders(config, use_subject_split, use_dolos_fold)

    # Build model
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


def train(config, use_subject_split=False, use_dolos_fold=False):
    """
    Training completo del modello.

    Args:
        config: Config object
        use_subject_split: Se True, usa subject-independent split custom
        use_dolos_fold: se True, usa dolos split ufficiale
    """
    print("\n" + "="*70)
    print("TRAINING COMPLETO")
    print("="*70)

    # Crea dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, use_subject_split, use_dolos_fold)

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


def test(config, use_dolos_fold=False):
    """
    Test finale con best model su test set.

    Args:
        config: Config object
        use_dolos_fold: Se True, usa test fold DOLOS
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING - Valutazione Finale su Test Set")
    print("=" * 70)

    # 1. Verifica che esista best model
    best_model_path = Path(config.get('paths.models_dir')) / 'best_model.pth'

    if not best_model_path.exists():
        print(f"❌ Best model non trovato: {best_model_path}")
        print("   Esegui prima il training!")
        return

    print(f"\n📂 Caricando best model da: {best_model_path}")

    # 2. Carica test set
    frames_dir = config.get('paths.frames_dir')
    annotation_file = config.get('paths.train_annotations')
    max_frames = 50
    sampling_strategy = config.get('preprocessing.sampling_strategy', 'uniform')

    if use_dolos_fold:
        # Test fold DOLOS
        fold_idx = config.get('dataset.fold_idx', 1)
        test_fold_path = Path(f'data/splits/test_fold{fold_idx}.csv')

        print(f"\n📊 Caricando test fold DOLOS: {test_fold_path.name}")

        test_clips = load_dolos_fold(test_fold_path)

        # Filtra clip disponibili
        full_dataset = DOLOSFrameDataset(
            root_dir=frames_dir,
            annotation_file=annotation_file,
            clip_filter=None,
            max_frames=max_frames
        )

        available_clips = set(s['clip_name'] for s in full_dataset.samples)
        test_clips_available = [c for c in test_clips if c in available_clips]

        print(f"   Clip disponibili: {len(test_clips_available)}/{len(test_clips)}")

        test_dataset = DOLOSFrameDataset(
            root_dir=frames_dir,
            annotation_file=annotation_file,
            clip_filter=set(test_clips_available),
            max_frames=max_frames
        )
    else:

        print(f"\n📊 Ricostruendo test set (random split)")

        _, _, test_dataset = create_random_split(
            frames_dir=frames_dir,
            annotation_file=annotation_file,
            train_ratio=config.get('dataset.train_ratio', 0.7),
            val_ratio=config.get('dataset.val_ratio', 0.15),
            test_ratio=config.get('dataset.test_ratio', 0.15),
            seed=config.get('dataset.seed', 42),
            max_frames=max_frames
        )

    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('testing.batch_size', 16),
        shuffle=False,
        collate_fn=collate_fn_with_padding,
        num_workers=config.get('training.num_workers', 2),
        pin_memory=config.get('training.pin_memory', True)
    )

    print(f"✅ Test set caricato: {len(test_dataset)} clips")

    # 3. Carica modello
    device = torch.device(config['training']['device'])
    model, _ = build_model(config)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Best model caricato (epoch {checkpoint['epoch'] + 1})")

    # 4. Inference su test set
    print(f"\n🔬 Eseguendo inference su test set...")

    metrics_tracker = MetricsTracker(
        num_classes=config.get('model.classifier.num_classes', 2),
        class_names=['Truth', 'Deception']
    )

    metrics_tracker.reset_epoch()

    with torch.no_grad():
        for frames, labels, lengths, mask, metadata in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            # Forward pass
            logits, _ = model(frames, mask)

            # Predictions
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # Update metrics
            metrics_tracker.update(
                predictions=predictions,
                targets=labels,
                loss=0.0,  # No loss calculation needed for test
                probabilities=probabilities
            )

    # 5. Calcola metriche finali
    test_metrics = metrics_tracker.compute_epoch_metrics('test')

    # 6. Print risultati
    print("\n" + "=" * 70)
    print("📊 TEST SET RESULTS - FINAL PERFORMANCE")
    print("=" * 70)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")

    if 'roc_auc' in test_metrics:
        print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    print("=" * 70)

    # Timestamp per organizzare i risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = Path(config.get('paths.results_dir'))
    test_results_dir = results_dir / 'test_results'
    test_results_dir.mkdir(parents=True, exist_ok=True)
    test_results_subdir = test_results_dir / f"run_{timestamp}"
    test_results_subdir.mkdir(parents=True, exist_ok=True)

    # 7. Classification report
    print("\n")
    metrics_tracker.generate_classification_report(save_dir=test_results_subdir)

    # 8. Salva risultati

    # Salva metriche

    test_results = {
        'test_metrics': test_metrics,
        'checkpoint_epoch': checkpoint['epoch'] + 1,
        'num_test_samples': len(test_dataset),
        'timestamp': datetime.now().isoformat()
    }

    with open(test_results_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✅ Risultati salvati in: {test_results_dir}")

    # 9. Genera grafici
    visualizer = MetricsVisualizer()

    # Confusion matrix
    visualizer.plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        class_names=['Truth', 'Deception'],
        save_dir=test_results_dir,
        normalize=False,
        show=False
    )

    visualizer.plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        class_names=['Truth', 'Deception'],
        save_dir=test_results_dir,
        normalize=True,
        show=False
    )

    # ROC curve
    if 'roc_curve' in test_metrics:
        visualizer.plot_roc_curve(
            test_metrics['roc_curve'],
            save_dir=test_results_dir,
            show=False
        )

    # Per-class metrics
    visualizer.plot_per_class_metrics(
        test_metrics['per_class'],
        class_names=['Truth', 'Deception'],
        save_dir=test_results_dir,
        show=False
    )

    print(f"✅ Grafici salvati in: {test_results_dir}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETATO!")
    print("=" * 70)
    print(f"🎯 Test F1: {test_metrics['f1']:.4f}")
    print(f"📁 Risultati: {test_results_dir}")
    print("=" * 70)

    return test_metrics


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

        # Prima testa con batch reale
        test_with_real_batch(config, args.subject_split, args.dolos_fold)

        # Poi training completo
        train(config, args.subject_split, args.dolos_fold)

    elif args.mode == 'test':
        test(config, use_dolos_fold=args.dolos_fold)

    print("\n" + "="*70)
    print("✓ COMPLETATO")
    print("="*70)


if __name__ == "__main__":
    main()