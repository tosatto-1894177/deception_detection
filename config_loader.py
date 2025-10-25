import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
import random


class Config:
    """
    Classe per caricare e gestire configurazioni da file YAML.
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path al file YAML di configurazione
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file non trovato: {config_path}")

        # Carica YAML
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Valida configurazione
        self._validate()

        # Setup riproducibilità
        self._setup_reproducibility()

        print(f"✓ Configurazione caricata da: {self.config_path}")

    def _validate(self):
        """Valida che le chiavi essenziali esistano."""
        required_sections = ['paths', 'preprocessing', 'dataset', 'model', 'training']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Sezione '{section}' mancante in config.yaml")

    def _setup_reproducibility(self):
        """Setup seed per riproducibilità."""
        if 'reproducibility' in self._config:
            seed = self._config['reproducibility']['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if self._config['reproducibility'].get('deterministic', False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                print(f"✓ Seed impostato a {seed} (modalità deterministica)")
            else:
                torch.backends.cudnn.benchmark = self._config['reproducibility'].get('benchmark', True)
                print(f"✓ Seed impostato a {seed}")

    def get(self, key: str, default=None) -> Any:
        """
        Accesso a configurazioni usando notazione dot.
        Esempio: config.get('paths.video_dir')
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Permette accesso dict-style: config['paths']"""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Permette: 'paths' in config"""
        return key in self._config

    def to_dict(self) -> Dict:
        """Ritorna configurazione come dizionario."""
        return self._config.copy()

    def save(self, save_path: str):
        """Salva configurazione in un nuovo file YAML."""
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Configurazione salvata in: {save_path}")

    def update(self, updates: Dict):
        """
        Aggiorna configurazione con nuovi valori.

        Args:
            updates: Dizionario con aggiornamenti (supporta nested dict)
        """

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self._config = deep_update(self._config, updates)

    def print_config(self, section: str = None):
        """
        Stampa configurazione (o una sezione specifica).

        Args:
            section: Nome della sezione da stampare (None = tutto)
        """
        import json

        if section:
            if section in self._config:
                print(f"\n{'=' * 50}")
                print(f"CONFIGURAZIONE - {section.upper()}")
                print(f"{'=' * 50}")
                print(json.dumps(self._config[section], indent=2))
            else:
                print(f"Sezione '{section}' non trovata")
        else:
            print(f"\n{'=' * 50}")
            print("CONFIGURAZIONE COMPLETA")
            print(f"{'=' * 50}")
            print(json.dumps(self._config, indent=2))

    def create_directories(self):
        """Crea tutte le directory specificate in paths."""
        paths_config = self._config.get('paths', {})

        for key, path in paths_config.items():
            if path and ('dir' in key.lower() or key in ['models_dir', 'logs_dir', 'results_dir']):
                Path(path).mkdir(parents=True, exist_ok=True)
                print(f"✓ Directory creata/verificata: {path}")


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Helper function per caricare configurazione.

    Args:
        config_path: Path al file YAML

    Returns:
        Oggetto Config
    """
    return Config(config_path)