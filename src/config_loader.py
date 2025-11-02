"""
Config loader - Carica e gestisce configurazioni da YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
import random


class Config:
    """Gestisce configurazione da file YAML."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config non trovato: {config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self._validate()
        self._setup_reproducibility()

        print(f"✓ Config caricato: {self.config_path}")

    def _validate(self):
        """Verifica sezioni essenziali."""
        required = ['paths', 'preprocessing', 'dataset', 'model', 'training']
        for section in required:
            if section not in self._config:
                raise ValueError(f"Sezione '{section}' mancante in config")

    def _setup_reproducibility(self):
        """Setup seed per riproducibilità."""
        if 'reproducibility' not in self._config:
            return

        seed = self._config['reproducibility']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self._config['reproducibility'].get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = self._config['reproducibility'].get('benchmark', True)

        print(f"✓ Seed: {seed}")

    def get(self, key: str, default=None) -> Any:
        """
        Accesso con notazione dot: config.get('paths.video_dir')
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
        """Accesso dict-style: config['paths']"""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def to_dict(self) -> Dict:
        return self._config.copy()

    def update(self, updates: Dict):
        """Aggiorna config con nuovi valori."""

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self._config = deep_update(self._config, updates)

    def save(self, save_path: str):
        """Salva config in nuovo file."""
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Config salvato: {save_path}")

    def create_directories(self):
        """Crea tutte le directory in paths."""
        paths = self._config.get('paths', {})

        for key, path in paths.items():
            if path and ('dir' in key.lower()):
                Path(path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config/config.yaml") -> Config:
    """Helper per caricare config."""
    return Config(config_path)