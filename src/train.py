"""
Training Loop completo con metriche
Sistema completo di training con early stopping, checkpointing e logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from src.model import build_model
from src.metrics import MetricsTracker, MetricsVisualizer


class Trainer:
    """
    Trainer completo per DcDtModel
    Gestisce training, validation, early stopping, checkpointing
    """

    def __init__(self, config, train_loader, val_loader=None):
        """
        Inizializza tutti i componenti necessari per il training

        Args:
            config: config.yaml
            train_loader: DataLoader training
            val_loader: DataLoader validation (opzionale)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(config['training']['device'])

        # Modello
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
        """Setta la loss function"""
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
        """Setta l'optimizer che si occupa di aggiornare i pesi"""
        opt_name = self.config.get('training.optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']


        # Solo parametri trainable
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_name == 'adam':
            weight_decay = self.config.get('training.weight_decay', 0.0001)
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            weight_decay = self.config.get('training.weight_decay', 0.01)
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer '{opt_name}' non supportato")

        print(f"Optimizer: {opt_name.upper()} (LR={lr}, WD={weight_decay})")
        return optimizer

    def _setup_scheduler(self):
        """Setta lo scheduler per ridurre il learning rate durante il training"""
        scheduler_config = self.config.get('training.scheduler')

        if scheduler_config is None or scheduler_config.get('type') is None:
            print("No LR scheduler")
            return None

        scheduler_type = scheduler_config['type'].lower()

        if scheduler_type == 'step':
            # Riduce LR ogni n epoch
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.5)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            print(f"LR Scheduler: StepLR (step={step_size}, gamma={gamma})")

        elif scheduler_type == 'cosine':
            # Riduzione "morbida" come funzione coseno
            T_max = self.config['training']['num_epochs']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
            print(f"LR Scheduler: CosineAnnealingLR (T_max={T_max})")

        elif scheduler_type == 'plateau':
            # Riduce quando validation non migliora
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
        """Training per una singola epoch"""
        self.model.train()
        self.metrics_tracker.reset_epoch()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [TRAIN]")

        for batch_idx, (frames, labels, lengths, mask, metadata) in enumerate(pbar):
            # Muove su device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            logits, attention_weights = self.model(frames, mask)

            # Calcola loss
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

            # Calcola metriche batch
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # Accumula metriche
            self.metrics_tracker.update(
                predictions=predictions,
                targets=labels,
                loss=loss.item(),
                probabilities=probabilities
            )

            # Aggiorna progress bar
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
        """Validation per una singola epoch"""
        if self.val_loader is None:
            return None

        self.model.eval()
        self.metrics_tracker.reset_epoch()

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [VAL]")

        for frames, labels, lengths, mask, metadata in pbar:
            # Muove su device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            logits, attention_weights = self.model(frames, mask)

            # Loss
            loss = self.criterion(logits, labels)

            # Calcola metriche batch
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # Accumula metriche
            self.metrics_tracker.update(
                predictions=predictions,
                targets=labels,
                loss=loss.item(),
                probabilities=probabilities
            )

            # Aggiorna progress bar
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
        Training loop completo

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

            # Stampa le metriche dell'epoch
            self.metrics_tracker.print_epoch_summary(
                epoch, metrics_train, metrics_val
            )

            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics_val['f1'] if metrics_val else metrics_train['f1'])
                else:
                    self.scheduler.step()

            # Gestione dei checkpoint
            is_best = False
            if metrics_val and metrics_val['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics_val['f1']
                is_best = True

            # Salva checkpoint ogni N epochs o se è il migliore
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
        print(f"Miglior valore F1: {self.best_val_f1:.4f}")
        print("=" * 70 + "\n")

        # Salva metriche e genera grafici
        self._save_final_results()

    def _save_final_results(self):
        """Salva risultati finali e genera grafici"""
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
    """Early stopping per evitare overfitting"""

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
        Aggiorna stato early stopping

        Args:
            score: Metrica da monitorare (es. val_f1)
        """
        if not self.enabled:
            return
        # Prima epoch
        if self.best_score is None:
            self.best_score = score
        # Non c'è miglioramento
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # Il counter ha raggiunto il limite, interrompi esecuzione
            if self.counter >= self.patience:
                self.should_stop = True
        # C'è miglioramento
        else:
            self.best_score = score
            self.counter = 0

def train_model(config, train_loader, val_loader=None):
    """
    Funzione wrapper per training da main.py

    Args:
        config: config.yaml
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (opzionale)

    Returns:
        trainer: Oggetto Trainer con modello addestrato
    """
    trainer = Trainer(config, train_loader, val_loader)
    trainer.train()

    return trainer