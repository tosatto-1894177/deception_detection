"""
Training Loop Completo con Metriche
Sistema completo di training con early stopping, checkpointing e logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from src.model import build_model
from src.frame_dataset import DOLOSFrameDataset, collate_fn_with_padding
from src.metrics import MetricsTracker, MetricsVisualizer


class Trainer:
    """
    Trainer completo per DeceptionNet.
    Gestisce training, validation, early stopping, checkpointing.
    """

    def __init__(self, config, train_loader, val_loader=None):
        """
        Args:
            config: Oggetto Config
            train_loader: DataLoader training
            val_loader: DataLoader validation (opzionale)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(config['training']['device'])
        print(f"Using device: {self.device}")

        # Build model
        self.model, _ = build_model(config)

        # Loss function
        self.criterion = self._setup_loss()

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler
        self.scheduler = self._setup_scheduler()

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            num_classes=config.get('model.classifier.num_classes', 2),
            class_names=['Truth', 'Deception']
        )

        # Visualizer
        self.visualizer = MetricsVisualizer()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('training.early_stopping.patience', 10),
            min_delta=config.get('training.early_stopping.min_delta', 0.001),
            enabled=config.get('training.early_stopping.enabled', True)
        )

        # Directories
        self.checkpoint_dir = Path(config.get('paths.models_dir'))
        self.results_dir = Path(config.get('paths.results_dir'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0

    def _setup_loss(self):
        """Setup loss function."""
        loss_type = self.config.get('training.loss.type', 'cross_entropy')

        if loss_type == 'cross_entropy':
            # Class weights per bilanciare classi
            class_weights = self.config.get('training.loss.class_weights')

            if class_weights is not None:
                weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
                print(f"Using CrossEntropyLoss with class weights: {class_weights}")
            else:
                criterion = nn.CrossEntropyLoss()
                print("Using CrossEntropyLoss (no class weights)")
        else:
            raise ValueError(f"Loss type '{loss_type}' non supportato")

        return criterion

    def _setup_optimizer(self):
        """Setup optimizer."""
        opt_name = self.config.get('training.optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config.get('training.weight_decay', 0.0)

        # Solo parametri trainable
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_name == 'adam':
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            optimizer = optim.SGD(trainable_params, lr=lr,
                                  momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer '{opt_name}' non supportato")

        print(f"Optimizer: {opt_name.upper()} (LR={lr}, WD={weight_decay})")
        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('training.scheduler')

        if scheduler_config is None or scheduler_config.get('type') is None:
            print("No LR scheduler")
            return None

        scheduler_type = scheduler_config['type'].lower()

        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.5)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            print(f"LR Scheduler: StepLR (step={step_size}, gamma={gamma})")

        elif scheduler_type == 'cosine':
            T_max = self.config['training']['num_epochs']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
            print(f"LR Scheduler: CosineAnnealingLR (T_max={T_max})")

        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5,
                patience=patience, verbose=True
            )
            print(f"LR Scheduler: ReduceLROnPlateau (patience={patience})")

        else:
            raise ValueError(f"Scheduler '{scheduler_type}' non supportato")

        return scheduler

    def train_epoch(self):
        """Training per una singola epoch."""
        self.model.train()
        self.metrics_tracker.reset_epoch()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [TRAIN]")

        for batch_idx, (frames, labels, lengths, mask, metadata) in enumerate(pbar):
            # Move to device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            logits, attention_weights = self.model(frames, mask)

            # Loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('training.gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            # Metrics
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            self.metrics_tracker.update(
                predictions=predictions,
                targets=labels,
                loss=loss.item(),
                probabilities=probabilities
            )

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(predictions == labels).float().mean().item():.4f}"
            })

            self.global_step += 1

        # Compute epoch metrics
        metrics = self.metrics_tracker.compute_epoch_metrics('train')

        return metrics

    @torch.no_grad()
    def validate_epoch(self):
        """Validation per una singola epoch."""
        if self.val_loader is None:
            return None

        self.model.eval()
        self.metrics_tracker.reset_epoch()

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [VAL]")

        for frames, labels, lengths, mask, metadata in pbar:
            # Move to device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            logits, attention_weights = self.model(frames, mask)

            # Loss
            loss = self.criterion(logits, labels)

            # Metrics
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            self.metrics_tracker.update(
                predictions=predictions,
                targets=labels,
                loss=loss.item(),
                probabilities=probabilities
            )

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(predictions == labels).float().mean().item():.4f}"
            })

        # Compute epoch metrics
        metrics = self.metrics_tracker.compute_epoch_metrics('val')

        return metrics

    def save_checkpoint(self, metrics_val=None, is_best=False):
        """
        Salva checkpoint del modello.

        Args:
            metrics_val: Metriche validation (opzionale)
            is_best: Se True, salva come best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics_val,
            'config': self.config.to_dict()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Salva checkpoint epoch corrente
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Salva best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model salvato: {best_path}")

        # Salva ultimo checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)

    def train(self, num_epochs=None):
        """
        Training loop completo.

        Args:
            num_epochs: Numero epoch (override config)
        """
        num_epochs = num_epochs or self.config['training']['num_epochs']

        print("\n" + "=" * 70)
        print("INIZIO TRAINING")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print("=" * 70 + "\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training
            metrics_train = self.train_epoch()

            # Validation
            metrics_val = self.validate_epoch()

            # Print summary
            self.metrics_tracker.print_epoch_summary(
                epoch, metrics_train, metrics_val
            )

            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics_val['f1'] if metrics_val else metrics_train['f1'])
                else:
                    self.scheduler.step()

            # Save checkpoint
            is_best = False
            if metrics_val and metrics_val['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics_val['f1']
                is_best = True

            # Save every N epochs or if best
            save_every = self.config.get('validation.save_every_n_epochs', 5)
            if is_best or (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                self.save_checkpoint(metrics_val, is_best)

            # Early stopping
            if metrics_val and self.early_stopping.enabled:
                self.early_stopping(metrics_val['f1'])

                if self.early_stopping.should_stop:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                    print(f"   No improvement for {self.early_stopping.patience} epochs")
                    break

        # Training completato
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETATO!")
        print("=" * 70)
        print(f"Tempo totale: {elapsed_time / 3600:.2f} ore")
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print("=" * 70 + "\n")

        # Salva metriche e genera grafici
        self._save_final_results()

    def _save_final_results(self):
        """Salva risultati finali e genera grafici."""
        print("\n" + "=" * 70)
        print("SALVATAGGIO RISULTATI FINALI")
        print("=" * 70 + "\n")

        # Timestamp per organizzare i risultati
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_subdir = self.results_dir / f"run_{timestamp}"
        results_subdir.mkdir(parents=True, exist_ok=True)

        # Salva metrics history
        self.metrics_tracker.save_metrics(results_subdir)

        # Genera classification report
        self.metrics_tracker.generate_classification_report(results_subdir)

        # Genera tutti i grafici
        self.visualizer.generate_all_plots(
            self.metrics_tracker,
            results_subdir,
            show=False
        )

        print(f"\n📁 Tutti i risultati salvati in: {results_subdir.resolve()}")
        print("\n✅ Pronto per la tesi! 🎓")


class EarlyStopping:
    """Early stopping per evitare overfitting."""

    def __init__(self, patience=10, min_delta=0.001, enabled=True):
        """
        Args:
            patience: Numero epoch da aspettare senza miglioramento
            min_delta: Miglioramento minimo considerato significativo
            enabled: Attiva/disattiva early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score):
        """
        Aggiorna stato early stopping.

        Args:
            score: Metrica da monitorare (es. val_f1)
        """
        if not self.enabled:
            return

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ============ FUNZIONE HELPER PER MAIN.PY ============

def train_model(config, train_loader, val_loader=None):
    """
    Funzione wrapper per training da main.py

    Args:
        config: Config object
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (opzionale)

    Returns:
        trainer: Oggetto Trainer con modello trainato
    """
    trainer = Trainer(config, train_loader, val_loader)
    trainer.train()

    return trainer


if __name__ == "__main__":
    """
    Test standalone del training loop.
    Usa dati fake per verificare che tutto funzioni.
    """
    print("=" * 70)
    print("TEST TRAINING LOOP")
    print("=" * 70)


    # Mock config
    class MockConfig:
        def __init__(self):
            self._config = {
                'model': {
                    'resnet': {
                        'architecture': 'resnet34',
                        'pretrained': True,
                        'freeze_layers': True
                    },
                    'tcn': {
                        'hidden_channels': [128, 128],
                        'kernel_size': 3,
                        'dropout': 0.2
                    },
                    'attention': {
                        'hidden_dim': 64
                    },
                    'classifier': {
                        'hidden_dims': [64],
                        'num_classes': 2,
                        'dropout': 0.3
                    }
                },
                'training': {
                    'device': 'cpu',
                    'num_epochs': 3,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0001,
                    'optimizer': 'adam',
                    'batch_size': 2,
                    'gradient_clip': 1.0,
                    'loss': {
                        'type': 'cross_entropy',
                        'class_weights': None
                    },
                    'scheduler': {
                        'type': 'step',
                        'step_size': 2,
                        'gamma': 0.5
                    },
                    'early_stopping': {
                        'enabled': True,
                        'patience': 5,
                        'min_delta': 0.001
                    }
                },
                'validation': {
                    'save_every_n_epochs': 1
                },
                'paths': {
                    'models_dir': 'models/checkpoints_test',
                    'results_dir': 'results/test_training'
                }
            }

        def get(self, key, default=None):
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

        def __getitem__(self, key):
            return self._config[key]

        def to_dict(self):
            return self._config.copy()


    # Mock dataset
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=20):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            seq_len = torch.randint(5, 15, (1,)).item()
            frames = torch.randn(seq_len, 3, 224, 224)
            label = idx % 2  # Alterna truth/lie

            return {
                'frames': frames,
                'label': label,
                'length': seq_len,
                'clip_name': f'clip_{idx}',
                'gender': 'Unknown'
            }


    from src.frame_dataset import collate_fn_with_padding

    # Create dataloaders
    train_dataset = MockDataset(20)
    val_dataset = MockDataset(10)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn_with_padding
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, collate_fn=collate_fn_with_padding
    )

    # Train
    config = MockConfig()
    trainer = train_model(config, train_loader, val_loader)

    print("\n✅ Test training loop completato!")
    print(f"Controlla 'results/test_training/' per i risultati")