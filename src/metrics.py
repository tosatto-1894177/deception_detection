"""
Metrics Tracking & Visualization
Sistema completo per tracciare metriche durante training e generare grafici finali
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from pathlib import Path
import json
from datetime import datetime


class MetricsTracker:
    """
    Tracker completo per metriche di training/validation/test
    Calcola e salva: F1, Accuracy, Precision, Recall, Loss, Confusion Matrix
    """

    def __init__(self, num_classes=2, class_names=None):
        """
        Args:
            num_classes: Numero di classi (2 per truth/lie)
            class_names: Lista dei nomi delle classi ['Truth', 'Deception']
        """
        self.num_classes = num_classes
        self.class_names = class_names or ['Truth', 'Deception']

        # Storia per ogni epoch
        self.history = {
            'train': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
            'val': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
        }

        # Migliori metriche
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Reset per epoch corrente
        self.reset_epoch()

    def reset_epoch(self):
        """Resetta accumulatori per nuova epoch"""
        self.epoch_predictions = []
        self.epoch_targets = []
        self.epoch_probabilities = []
        self.epoch_losses = []

    def update(self, predictions, targets, loss=None, probabilities=None):
        """
        Aggiorna metriche con batch corrente

        Args:
            predictions: (batch_size,) tensor o list di predizioni
            targets: (batch_size,) tensor o list di target reali
            loss: scalar, loss del batch (opzionale)
            probabilities: (batch_size, num_classes) probabilità (opzionale)
        """
        # Converti a numpy (prima muove su cpu)
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        if torch.is_tensor(probabilities):
            probabilities = probabilities.detach().cpu().numpy()

        self.epoch_predictions.extend(predictions.tolist())
        self.epoch_targets.extend(targets.tolist())

        if loss is not None:
            self.epoch_losses.append(loss)

        if probabilities is not None:
            self.epoch_probabilities.extend(probabilities.tolist())

    def compute_epoch_metrics(self, phase='train'):
        """
        Calcola metriche aggregate per l'epoch corrente

        Args:
            phase: 'train' o 'val'

        Returns:
            dict con tutte le metriche
        """
        # recupera le predizioni e i target dell'epoch
        predictions = np.array(self.epoch_predictions)
        targets = np.array(self.epoch_targets)

        # Calcola metriche
        metrics = {'loss': np.mean(self.epoch_losses) if self.epoch_losses else 0.0,
                   'accuracy': accuracy_score(targets, predictions),
                   'precision': precision_score(targets, predictions, average='binary', zero_division=0),
                   'recall': recall_score(targets, predictions, average='binary', zero_division=0),
                   'f1': f1_score(targets, predictions, average='binary', zero_division=0),
                   'confusion_matrix': confusion_matrix(targets, predictions)}

        # Metriche per classe
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)

        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist()
        }

        # ROC-AUC se abbiamo probabilità
        if self.epoch_probabilities:
            probabilities = np.array(self.epoch_probabilities)
            if probabilities.shape[1] == self.num_classes:
                # Usa probabilità classe positiva (deception = classe 1)
                fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

        # Salva in history se train o val
        if phase in self.history:
            self.history[phase]['loss'].append(metrics['loss'])
            self.history[phase]['accuracy'].append(metrics['accuracy'])
            self.history[phase]['f1'].append(metrics['f1'])
            self.history[phase]['precision'].append(metrics['precision'])
            self.history[phase]['recall'].append(metrics['recall'])

        # Aggiorna le metriche migliori (solo per validation)
        if phase == 'val':
            if metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['f1']
                self.best_epoch = len(self.history['val']['f1']) - 1
            if metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = metrics['accuracy']

        return metrics

    def print_epoch_summary(self, epoch, metrics_train, metrics_val=None):
        """Stampa riassunto metriche per epoch"""
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'=' * 70}")

        # Training metrics
        print(f"\n📊 TRAINING:")
        print(f"  Loss:      {metrics_train['loss']:.4f}")
        print(f"  Accuracy:  {metrics_train['accuracy']:.4f}")
        print(f"  Precision: {metrics_train['precision']:.4f}")
        print(f"  Recall:    {metrics_train['recall']:.4f}")
        print(f"  F1 Score:  {metrics_train['f1']:.4f}")

        # Validation metrics
        if metrics_val:
            print(f"\n📊 VALIDATION:")
            print(f"  Loss:      {metrics_val['loss']:.4f}")
            print(f"  Accuracy:  {metrics_val['accuracy']:.4f}")
            print(f"  Precision: {metrics_val['precision']:.4f}")
            print(f"  Recall:    {metrics_val['recall']:.4f}")
            print(f"  F1 Score:  {metrics_val['f1']:.4f}")

            if 'roc_auc' in metrics_val:
                print(f"  ROC-AUC:   {metrics_val['roc_auc']:.4f}")

            print(f"\n🏆 Best Val F1: {self.best_val_f1:.4f} (Epoch {self.best_epoch + 1})")

        print(f"{'=' * 70}\n")

    def save_metrics(self, save_dir, filename='metrics_history.json'):
        """Salva la metric history in uno JSON"""
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_dict = {
            'history': self.history,
            'best_val_f1': float(self.best_val_f1),
            'best_val_acc': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch),
            'timestamp': datetime.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"✓ Metrics salvate in: {save_path}")

    def save_confusion_matrix(self, save_dir, phase='val', filename='confusion_matrix_final.npy'):
        """Salva confusion matrix finale"""
        save_path = Path(save_dir) / filename

        predictions = np.array(self.epoch_predictions)
        targets = np.array(self.epoch_targets)
        cm = confusion_matrix(targets, predictions)

        np.save(save_path, cm)
        print(f"✓ Confusion matrix salvata in: {save_path}")

        return cm

    def generate_classification_report(self, save_dir=None):
        """Genera classification report dettagliato"""
        predictions = np.array(self.epoch_predictions)
        targets = np.array(self.epoch_targets)

        report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            digits=4
        )

        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(report)

        if save_dir:
            save_path = Path(save_dir) / 'classification_report.txt'
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✓ Report salvato in: {save_path}")

        return report


class MetricsVisualizer:
    """
    Crea grafici delle metriche
    """

    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Args:
            style: Stile matplotlib
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.colors = {
            'train': '#2E86AB',  # Blu
            'val': '#A23B72',  # Viola
            'truth': '#06A77D',  # Verde
            'lie': '#D74E09'  # Rosso
        }

    def plot_training_curves(self, history, save_dir, show=False):
        """
        Plotta curve di training (loss, accuracy, F1, precision, recall)

        Args:
            history: Dict da MetricsTracker.history
            save_dir: Directory dove salvare i plot
            show: Se True, mostra i plot interattivi
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall']
        titles = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            epochs = range(1, len(history['train'][metric]) + 1)

            # Training curve
            ax.plot(epochs, history['train'][metric],
                    label='Training', marker='o', linewidth=2,
                    color=self.colors['train'])

            # Validation curve (se esiste)
            if history['val'][metric]:
                ax.plot(epochs, history['val'][metric],
                        label='Validation', marker='s', linewidth=2,
                        color=self.colors['val'])

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} over Epochs', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        # Rimuovi subplot extra
        fig.delaxes(axes[5])

        plt.tight_layout()

        # Salva
        save_path = save_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves salvate in: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confusion_matrix(self, cm, class_names, save_dir,
                              normalize=False, show=False):
        """
        Plotta confusion matrix

        Args:
            cm: Confusion matrix (2x2 array)
            class_names: Lista dei nomi delle classi
            save_dir: Directory output
            normalize: Se True, normalizza per riga (recall)
            show: Mostra plot interattivo
        """
        save_dir = Path(save_dir)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
            filename = 'confusion_matrix_normalized.png'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            filename = 'confusion_matrix.png'

        fig, ax = plt.subplots(figsize=(8, 6))

        # Heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                    linewidths=2, linecolor='white', ax=ax)

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        save_path = save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix salvata in: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_roc_curve(self, roc_data, save_dir, show=False):
        """
        Plotta ROC curve

        Args:
            roc_data: Dict con 'fpr', 'tpr' da MetricsTracker
            save_dir: Directory output
            show: Mostra plot
        """
        save_dir = Path(save_dir)

        fpr = np.array(roc_data['fpr'])
        tpr = np.array(roc_data['tpr'])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        # ROC curve
        ax.plot(fpr, tpr, color=self.colors['val'], linewidth=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')

        # Diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2,
                label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = save_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve salvata in: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_per_class_metrics(self, metrics_per_class, class_names,
                               save_dir, show=False):
        """
        Plotta metriche per classe (bar chart)

        Args:
            metrics_per_class: Dict con 'precision', 'recall', 'f1' per classe
            class_names: Lista nomi classi
            save_dir: Directory output
            show: Mostra plot
        """
        save_dir = Path(save_dir)

        metrics = ['precision', 'recall', 'f1']
        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, metric in enumerate(metrics):
            values = metrics_per_class[metric]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=metric.capitalize())

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        plt.tight_layout()

        save_path = save_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics salvate in: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_all_plots(self, tracker, save_dir, show=False):
        """
        Genera tutti i plot in una volta

        Args:
            tracker: Istanza di MetricsTracker
            save_dir: Directory output
            show: Mostra plot interattivi
        """
        print("\n" + "=" * 70)
        print("GENERAZIONE GRAFICI")
        print("=" * 70 + "\n")

        save_dir = Path(save_dir)

        # 1. Training curves
        self.plot_training_curves(tracker.history, save_dir, show)

        # 2. Confusion matrix
        predictions = np.array(tracker.epoch_predictions)
        targets = np.array(tracker.epoch_targets)
        cm = confusion_matrix(targets, predictions)

        # Non-normalized
        self.plot_confusion_matrix(cm, tracker.class_names, save_dir,
                                   normalize=False, show=show)

        # Normalized
        self.plot_confusion_matrix(cm, tracker.class_names, save_dir,
                                   normalize=True, show=show)

        # 3. ROC curve (se disponibile)
        if tracker.epoch_probabilities:
            probabilities = np.array(tracker.epoch_probabilities)
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            self.plot_roc_curve(roc_data, save_dir, show)

        # 4. Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)

        metrics_per_class = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist()
        }

        self.plot_per_class_metrics(metrics_per_class, tracker.class_names,
                                    save_dir, show)

        print("\n✅ Tutti i grafici generati con successo!")
        print(f"📁 Salvati in: {save_dir.resolve()}\n")