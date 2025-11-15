"""
Evaluation & Testing
Classe Evaluator per gestire test/validation finale su dataset
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.metrics import MetricsTracker, MetricsVisualizer


class Evaluator:
    """
    Evaluator per test finale del modello

    Gestisce:
    - Inference su test set
    - Calcolo metriche dettagliate
    - Salvataggio predictions, attention weights, analisi errori
    - Generazione grafici
    """

    def __init__(self, model, config, test_loader, device=None, save_results=True):
        """
        Args:
            model: Modello addestrato (DcDtModel)
            config: config.yaml
            test_loader: DataLoader test set
            device: device
            save_results: Se True, salva tutti i file (CSV, JSON, plot)
                         Se False, ritorna solo metriche (per demo mode)
        """
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.device = device or torch.device(config['training']['device'])
        self.save_results = save_results

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            num_classes=config.get('model.classifier.num_classes', 2),
            class_names=['Truth', 'Deception']
        )

        # Visualizer
        self.visualizer = MetricsVisualizer()

        # Results directory (se save_results=True)
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(config.get('paths.results_dir'))
            test_results_dir = results_dir / 'test_results'
            test_results_dir.mkdir(parents=True, exist_ok=True)
            self.results_subdir = test_results_dir / f"run_{timestamp}"
            self.results_subdir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_subdir = None

    def evaluate(self):
        """
        Esegue valutazione completa sul test set

        Returns:
            dict: Metriche test complete
        """
        print("\n" + "=" * 80)
        print("ESECUZIONE TEST")
        print("=" * 80)
        print(f"Samples di test: {len(self.test_loader.dataset)}")
        if self.save_results:
            print(f"Cartella di output: {self.results_subdir}")
        print("=" * 80 + "\n")

        self.model.eval()
        self.metrics_tracker.reset_epoch()

        # Liste per raccogliere dati dettagliati
        predictions_list = []
        attention_data = []

        # Contatori per clip "interessanti" - max 5 per classe
        num_correct_saved = 0
        num_incorrect_saved = 0
        MAX_SAMPLES = 5

        # Inference loop
        with torch.no_grad():
            for frames, labels, lengths, mask, metadata in tqdm(self.test_loader, desc="Testing"):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                logits, attention_weights = self.model(frames, mask)

                # Predictions
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)

                # Update metrics
                self.metrics_tracker.update(
                    predictions=predictions,
                    targets=labels,
                    loss=0.0,
                    probabilities=probabilities
                )

                # Salva predictions dettagliate per ogni clip
                for i in range(len(predictions)):
                    pred_dict = {
                        'clip_name': metadata['clip_names'][i],
                        'true_label': 'Truth' if labels[i].item() == 0 else 'Deception',
                        'predicted_label': 'Truth' if predictions[i].item() == 0 else 'Deception',
                        'confidence_truth': probabilities[i, 0].item(),
                        'confidence_deception': probabilities[i, 1].item(),
                        'num_frames': lengths[i].item(),
                        'correct': (predictions[i] == labels[i]).item()
                    }
                    predictions_list.append(pred_dict)

                    if not attention_complete:
                        is_correct = (predictions[i] == labels[i]).item()

                        if is_correct and num_correct_saved < MAX_SAMPLES:
                            attention_data.append({
                                'clip_name': metadata['clip_names'][i],
                                'attention_weights': attention_weights[i, :lengths[i]].cpu().numpy().tolist(),
                                'num_frames': lengths[i].item(),
                                'true_label': 'Truth' if labels[i].item() == 0 else 'Deception',
                                'predicted_label': 'Truth' if predictions[i].item() == 0 else 'Deception',
                                'correct': True
                            })
                            num_correct_saved += 1

                        elif not is_correct and num_incorrect_saved < MAX_SAMPLES:
                            attention_data.append({
                                'clip_name': metadata['clip_names'][i],
                                'attention_weights': attention_weights[i, :lengths[i]].cpu().numpy().tolist(),
                                'num_frames': lengths[i].item(),
                                'true_label': 'Truth' if labels[i].item() == 0 else 'Deception',
                                'predicted_label': 'Truth' if predictions[i].item() == 0 else 'Deception',
                                'correct': False
                            })
                            num_incorrect_saved += 1

                        if num_correct_saved >= MAX_SAMPLES and num_incorrect_saved >= MAX_SAMPLES:
                            attention_complete = True

        # Calcola metriche finali
        test_metrics = self.metrics_tracker.compute_epoch_metrics('test')

        # Analisi dettagliata predictions
        analysis_results = self._analyze_predictions(predictions_list)

        # Stampa risultati
        self._print_results(test_metrics)

        # Salva tutto se richiesto
        if self.save_results:
            self._save_all_results(
                test_metrics,
                predictions_list,
                attention_data,
                analysis_results
            )

        return test_metrics

    def _analyze_predictions(self, predictions_list):
        """
        Analizza le predictions per generare statistiche dettagliate

        Args:
            predictions_list: Lista dict con predictions per ogni clip

        Returns:
            dict: error_analysis, confidence_analysis, dataframes
        """
        df_predictions = pd.DataFrame(predictions_list)

        # A. Clip misclassificate
        misclassified = df_predictions[df_predictions['correct'] == False]

        # B. Analisi errori per classe
        truth_total = len(df_predictions[df_predictions['true_label'] == 'Truth'])
        lie_total = len(df_predictions[df_predictions['true_label'] == 'Deception'])
        truth_errors = len(misclassified[misclassified['true_label'] == 'Truth'])
        lie_errors = len(misclassified[misclassified['true_label'] == 'Deception'])

        error_analysis = {
            'truth_misclassified': int(truth_errors),
            'truth_total': int(truth_total),
            'truth_error_rate': float(truth_errors / truth_total if truth_total > 0 else 0),
            'deception_misclassified': int(lie_errors),
            'deception_total': int(lie_total),
            'deception_error_rate': float(lie_errors / lie_total if lie_total > 0 else 0)
        }

        # C. Analisi confidence
        correct_preds = df_predictions[df_predictions['correct'] == True]
        incorrect_preds = df_predictions[df_predictions['correct'] == False]

        # Confidence per predictions corrette
        correct_confidences = []
        for _, row in correct_preds.iterrows():
            if row['predicted_label'] == 'Truth':
                correct_confidences.append(row['confidence_truth'])
            else:
                correct_confidences.append(row['confidence_deception'])

        # Confidence per predictions sbagliate
        incorrect_confidences = []
        for _, row in incorrect_preds.iterrows():
            if row['predicted_label'] == 'Truth':
                incorrect_confidences.append(row['confidence_truth'])
            else:
                incorrect_confidences.append(row['confidence_deception'])

        confidence_analysis = {
            'correct_mean_confidence': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
            'incorrect_mean_confidence': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
            'correct_std_confidence': float(np.std(correct_confidences)) if correct_confidences else 0.0,
            'incorrect_std_confidence': float(np.std(incorrect_confidences)) if incorrect_confidences else 0.0
        }

        return {
            'df_predictions': df_predictions,
            'misclassified': misclassified,
            'error_analysis': error_analysis,
            'confidence_analysis': confidence_analysis,
            'num_correct': len(correct_preds),
            'num_incorrect': len(incorrect_preds)
        }

    def _print_results(self, test_metrics):
        """Stampa risultati test a schermo"""
        print("\n" + "=" * 80)
        print("RISULTATI DEL SET DI TEST - PERFORMANCE FINALE")
        print("=" * 80)
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print(f"F1 Score:  {test_metrics['f1']:.4f}")

        if 'roc_auc' in test_metrics:
            print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")

        print("=" * 80)

    def _save_all_results(self, test_metrics, predictions_list, attention_data, analysis_results):
        """
        Salva tutti i risultati: CSV, JSON, grafici

        Args:
            test_metrics: Dict metriche test
            predictions_list: Lista predictions per clip
            attention_data: Lista attention weights
            analysis_results: Dict con analisi dettagliate
        """
        print("\n" + "=" * 80)
        print("Salvataggio risultati di test")
        print("=" * 80 + "\n")

        # 1. CSV predictions dettagliate
        df_predictions = analysis_results['df_predictions']
        df_predictions.to_csv(self.results_subdir / 'predictions_detailed.csv', index=False)
        print(f"✅ Predictions salvate: predictions_detailed.csv")

        # 2. CSV misclassified
        misclassified = analysis_results['misclassified']
        misclassified.to_csv(self.results_subdir / 'misclassified.csv', index=False)
        print(f"✅ Misclassified salvate: misclassified.csv ({len(misclassified)} clip)")

        # 3. JSON attention weights
        with open(self.results_subdir / 'attention_weights_sample.json', 'w') as f:
            json.dump(attention_data, f, indent=2)
        print(f"✅ Attention weights salvate: attention_weights_sample.json ({len(attention_data)} clip)")

        # 4. JSON risultati completi
        test_metrics_json = self._convert_to_json_serializable(test_metrics)

        test_results = {
            'test_metrics': test_metrics_json,
            'num_test_samples': len(self.test_loader.dataset),
            'timestamp': datetime.now().isoformat(),

            # Analisi dettagliate
            'error_analysis': analysis_results['error_analysis'],
            'confidence_analysis': analysis_results['confidence_analysis'],

            # Info aggiuntive
            'num_correct': int(analysis_results['num_correct']),
            'num_incorrect': int(analysis_results['num_incorrect']),
            'num_misclassified_truth': int(analysis_results['error_analysis']['truth_misclassified']),
            'num_misclassified_deception': int(analysis_results['error_analysis']['deception_misclassified'])
        }

        with open(self.results_subdir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"✅ Test results salvati: test_results.json")

        # 5. Classification report
        self.metrics_tracker.generate_classification_report(self.results_subdir)

        # 6. Grafici
        print("\nGenerazione grafici...")

        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=['Truth', 'Deception'],
            save_dir=self.results_subdir,
            normalize=False,
            show=False
        )

        # Confusion matrix normalizzata
        self.visualizer.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=['Truth', 'Deception'],
            save_dir=self.results_subdir,
            normalize=True,
            show=False
        )

        # ROC curve
        if 'roc_curve' in test_metrics:
            self.visualizer.plot_roc_curve(
                test_metrics['roc_curve'],
                save_dir=self.results_subdir,
                show=False
            )

        # Per-class metrics
        self.visualizer.plot_per_class_metrics(
            test_metrics['per_class'],
            class_names=['Truth', 'Deception'],
            save_dir=self.results_subdir,
            show=False
        )

        print(f"✅ Grafici salvati in: {self.results_subdir}")

        print("\n" + "=" * 80)
        print("✅ TUTTI I RISULTATI SALVATI!")
        print("=" * 80)
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Risultati: {self.results_subdir}")
        print("=" * 80)

    def _convert_to_json_serializable(self, obj):
        """Converte numpy arrays in liste per JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def evaluate_model(model, config, test_loader, save_results=True, max_attention_samples=5):
    """
    Funzione wrapper per testing da main.py

    Args:
        model: Modello addestrato
        config: Configurazione
        test_loader: DataLoader test set
        save_results: Se True, salva tutti i file
        max_attention_samples: Max clip per attention weights

    Returns:
        dict: Metriche test
    """
    evaluator = Evaluator(
        model=model,
        config=config,
        test_loader=test_loader,
        save_results=save_results
    )

    test_metrics = evaluator.evaluate()

    return test_metrics