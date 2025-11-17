"""
Deception Detection - Main Pipeline
Gestisce preprocessing, training e test del modello
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.config_loader import load_config
from scripts.frame_extractor import process_dolos_with_faces
from src.dataset import DOLOSDataset, collate_fn_with_padding, load_dolos_fold
from src.train import train_model
from src.model import build_model
from torch.utils.data import Subset
import numpy as np
from src.evaluate import evaluate_model
from datetime import datetime

# Ottimizzazione memoria GPU
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


def parse_args():
    """Parsa gli argomenti della linea di comando"""
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

    # Subject split - se presente gestisce la creazione del dataset con subject split
    parser.add_argument('--subject_split', action='store_true',
                        help='Usa subject-independent split custom (per training finale)')

    # Split con fold DOLOS
    parser.add_argument('--dolos_fold', action='store_true',
                        help='Usa fold DOLOS ufficiali')

    return parser.parse_args()


def preprocess(config):
    """
    Modalità per l'estrazione dei frame dai video con face detection
    Utilizza Haar
    """
    print("\n" + "="*80)
    print("PREPROCESSING: Estrazione frame con face detection")
    print("="*80)

    video_dir = Path(config.get('paths.video_dir'))
    frames_dir = Path(config.get('paths.frames_dir'))

    print(f"  Cartella delle clip: {video_dir.resolve()}")

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
    Crea DataLoader per train/val/test

    Args:
        config: config.yaml
        use_subject_split: Se True utilizza la funzione per fare subject-independent split
        use_dolos_fold: Se True, utilizza i fold DOLOS ufficiali (non subject-independent)
        Se entrambi False, utilizza random split (più veloce per test)
    """
    print("\n" + "="*80)
    print("CARICAMENTO DATASET")
    print("="*80)

    frames_dir = config.get('paths.frames_dir')
    batch_size = config['training']['batch_size']
    annotation_file = config.get('paths.train_annotations')

    use_openface = config.get('dataset.use_openface', False)
    openface_csv_dir = config.get('paths.openface_csv_dir') if use_openface else None

    if use_dolos_fold:
        # Usa fold DOLOS ufficiali

        from src.dataset import create_dolos_fold_split

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
            max_frames=50,
            use_openface=use_openface,
            openface_csv_dir=openface_csv_dir
        )

    """elif use_subject_split:
        # Subject-Independent Split

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
        )"""

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
    Funzione di utility:
        Esegue un test con un batch reale dal dataset
        Verifica che tutto funzioni prima del training completo.
    """
    print("\n" + "="*80)
    print("TEST CON BATCH REALE")
    print("="*80)

    # Crea il dataloader
    train_loader, _, _ = create_dataloaders(config, use_subject_split, use_dolos_fold)

    # Costruisce il modello
    model, device = build_model(config)

    print("\nCaricamento primo batch")
    batch_data = next(iter(train_loader))

    # Unpack - gestisce sia 5 che 6 elementi
    if len(batch_data) == 6:
        frames, labels, lengths, mask, metadata, openface = batch_data
    else:
        frames, labels, lengths, mask, metadata = batch_data
        openface = None

    print(f"\nBatch info:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Labels: {labels}")
    print(f"  Lengths: {lengths}")
    print(f"  Clip names: {metadata['clip_names']}")
    if openface is not None:
        print(f"  OpenFace shape: {openface.shape}")
        openface = openface.to(device)
    else:
        print(f"  OpenFace: None")

    # Forward pass
    print("\nForward pass...")
    frames = frames.to(device)
    mask = mask.to(device)

    model.eval()
    with torch.no_grad():
        # Gestisce entrambi i tipi di modello
        is_multimodal = config.get('model.type', 'video_only') == 'multimodal'

        if is_multimodal:
            logits, attention_dict = model(frames, mask, openface)
            attention_weights = attention_dict['video']
        else:
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


def demo_mode(config, use_dolos_fold=False):
    """
    Modalità demo: mini-training su subset piccolo per test rapido.
    Utile per verificare che tutto funzioni prima del training completo.

    Args:
        config: config.yaml
        use_dolos_fold: Se True, usa fold DOLOS ufficiali
    """
    print("\n" + "="*80)
    print("DEMO MODE: Mini-training + test su subset piccolo per verifiche rapide.")
    print("="*80)

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
    print(f"  Split: {'DOLOS Fold' if use_dolos_fold else 'Random'}")

    # Crea dataloaders con RANDOM split
    train_loader, val_loader, test_loader = create_dataloaders(config, use_subject_split=False, use_dolos_fold=use_dolos_fold)

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

    print("\n" + "=" * 80)
    print("FASE 1: MINI-TRAINING")
    print("=" * 80)

    # Training
    trainer = train_model(config, train_loader, val_loader)

    print(f"\n✅ Training demo completato!")
    print(f"   Best val F1: {trainer.best_val_f1:.4f}")

    # Test
    print("\n" + "=" * 80)
    print("FASE 2: MINI-TEST")
    print("=" * 80)

    # Crea subset ridotto del test set
    test_indices = np.arange(min(10, len(test_loader.dataset)))
    test_subset = Subset(test_loader.dataset, test_indices)
    test_loader_mini = DataLoader(
        test_subset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn_with_padding
    )

    print(f"⚠️ Demo test: usando solo {len(test_subset)} clip test")

    # Carica best model e testa
    best_model_path = Path(config.get('paths.models_dir')) / 'best_model.pth'

    if best_model_path.exists():
        device = torch.device(config['training']['device'])
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✅ Best model caricato (epoch {checkpoint['epoch'] + 1})")

        test_metrics = evaluate_model(
            model=trainer.model,
            config=config,
            test_loader=test_loader_mini,
            save_results=True,
            max_attention_samples=3
        )

        print(f"\n✅ Mini-test completato!")
        print(f"   Test F1: {test_metrics['f1']:.4f}")
    else:
        print("⚠️ Best model non trovato, salto test")

    print(f"\n✅ Demo completato!")
    print(f"   Se tutto è OK, esegui training completo con --mode train")


def train(config, use_subject_split=False, use_dolos_fold=False):
    """
    Training completo del modello

    Args:
        config: config.yaml
        use_subject_split: Se True utilizza la funzione per fare subject-independent split
        use_dolos_fold: Se True, utilizza i fold DOLOS ufficiali (non subject-independent)
        Se entrambi False, utilizza random split (più veloce per test)
    """
    print("\n" + "="*80)
    print("TRAINING COMPLETO")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get('paths.models_dir')) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.update({'paths': {'models_dir': str(run_dir)}})

    # Crea dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, use_subject_split, use_dolos_fold)

    # Costruisce il trainer, ovvero costruisce il modello e lo addestra
    trainer = train_model(config, train_loader, val_loader)

    # Salva modello finale
    final_model_path = run_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config.to_dict(),
        'best_val_f1': trainer.best_val_f1
    }, final_model_path)

    print(f"\n✓ Modello finale salvato: {final_model_path}")

    return trainer


def test(config, use_dolos_fold=False):
    """
    Test finale tramite il modello migliore salvato durante training

    Args:
        config: config.yaml
        use_dolos_fold: Se True, usa test fold DOLOS
    """

    # 1. Verifica che esista best model salvato
    best_model_path = Path(config.get('paths.models_dir')) / 'best_model.pth'

    if not best_model_path.exists():
        print(f"❌ Modello migliore non trovato: {best_model_path}")
        print("   Esegui prima il training!")
        return

    print(f"\n📂 Caricando il modello migliore da: {best_model_path}")

    # 2. Carica test set
    frames_dir = config.get('paths.frames_dir')
    annotation_file = config.get('paths.train_annotations')
    max_frames = 50

    use_openface = config.get('dataset.use_openface', False)
    openface_csv_dir = config.get('paths.openface_csv_dir') if use_openface else None

    if use_dolos_fold:
        # Test fold DOLOS
        fold_idx = config.get('dataset.fold_idx', 1)
        test_fold_path = Path(f'data/splits/test_fold{fold_idx}.csv')

        test_clips = load_dolos_fold(test_fold_path)

        # Filtra clip disponibili
        full_dataset = DOLOSDataset(
            root_dir=frames_dir,
            annotation_file=annotation_file,
            clip_filter=None,
            max_frames=max_frames,
            use_openface=use_openface,
            openface_csv_dir=openface_csv_dir
        )

        available_clips = set(s['clip_name'] for s in full_dataset.samples)
        test_clips_available = [c for c in test_clips if c in available_clips]

        print(f"   Clip disponibili: {len(test_clips_available)}/{len(test_clips)}")

        test_dataset = DOLOSDataset(
            root_dir=frames_dir,
            annotation_file=annotation_file,
            clip_filter=set(test_clips_available),
            max_frames=max_frames,
            use_openface=use_openface,
            openface_csv_dir=openface_csv_dir
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

    print(f"✅ Set di test caricato: {len(test_dataset)} clips")

    # 3. Carica modello
    device = torch.device(config['training']['device'])
    model, _ = build_model(config)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Modello migliore caricato (epoch {checkpoint['epoch'] + 1})")

    # 4. Esegui evaluation
    test_metrics = evaluate_model(
        model=model,
        config=config,
        test_loader=test_loader,
        save_results=True,
        max_attention_samples=5
    )

    return test_metrics

def convert_to_json_serializable(obj):
    """Converte ricorsivamente numpy arrays in liste per JSON"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def main():
    """Entry point del main"""
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
        demo_mode(config, use_dolos_fold=args.dolos_fold)

    elif args.mode == 'train':

        # Prima testa con batch reale
        test_with_real_batch(config, args.subject_split, args.dolos_fold)

        # Poi training completo
        train(config, args.subject_split, args.dolos_fold)

    elif args.mode == 'test':
        test(config, use_dolos_fold=args.dolos_fold)

    print("\n" + "="*80)
    print("✓ COMPLETATO")
    print("="*80)


if __name__ == "__main__":
    main()