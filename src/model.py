"""
DeceptionNet: Modello completo per Deception Detection
Architettura: ResNet → TCN → Attention → MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights


class ResNetFeatureExtractor(nn.Module):
    """Wrapper per ResNet come feature extractor."""

    def __init__(self, architecture='resnet34', pretrained=True, freeze=True):
        super().__init__()

        if architecture == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.resnet = resnet34(weights=weights)
            self.feature_dim = 512
        elif architecture == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.resnet = resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Architecture '{architecture}' non supportata")

        # Rimuovi classificatore finale
        self.resnet.fc = nn.Identity()

        # Freeze layers se richiesto
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()

    def forward(self, x):
        """
        Args:
            x: (batch*seq_len, C, H, W)
        Returns:
            features: (batch*seq_len, feature_dim)
        """
        return self.resnet(x)


class TemporalConvBlock(nn.Module):
    """Singolo blocco convoluzionale temporale con residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()

        # Padding per mantenere lunghezza temporale
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = F.relu(out)
        out = self.dropout2(out)

        return out


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network per catturare dipendenze temporali."""

    def __init__(self, input_dim, hidden_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            input_dim: Dimensione feature input (512 per ResNet34)
            hidden_channels: Lista con numero canali per ogni layer [256, 256, 256]
            kernel_size: Dimensione kernel convoluzionale
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        num_levels = len(hidden_channels)

        for i in range(num_levels):
            in_ch = input_dim if i == 0 else hidden_channels[i - 1]
            out_ch = hidden_channels[i]

            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dropout))

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_channels[-1]

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, feature_dim)
            mask: (batch, seq_len) - True per frame reali, False per padding
        Returns:
            out: (batch, seq_len, output_dim)
        """
        # Conv1d richiede (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, feature_dim, seq_len)

        out = self.network(x)

        # Torna a (batch, seq_len, channels)
        out = out.transpose(1, 2)

        # Applica mask per azzerare padding
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class TemporalAttention(nn.Module):
    """Attention mechanism per aggregare frame temporali."""

    def __init__(self, input_dim, hidden_dim=128):
        """
        Args:
            input_dim: Dimensione feature input da TCN
            hidden_dim: Dimensione hidden layer attention
        """
        super().__init__()

        # Attention layers
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, feature_dim)
            mask: (batch, seq_len) - True per frame reali
        Returns:
            context: (batch, feature_dim) - feature aggregate
            attention_weights: (batch, seq_len) - pesi attention
        """
        # Calcola attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Applica mask: metti -inf dove c'è padding
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax per normalizzare
        attention_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), x)  # (batch, 1, feature_dim)
        context = context.squeeze(1)  # (batch, feature_dim)

        return context, attention_weights


class MLPClassifier(nn.Module):
    """MLP finale per classificazione."""

    def __init__(self, input_dim, hidden_dims, num_classes=2, dropout=0.3):
        """
        Args:
            input_dim: Dimensione input da attention
            hidden_dims: Lista hidden layers [128, 64]
            num_classes: Numero classi output (2 per truth/lie)
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, feature_dim)
        Returns:
            logits: (batch, num_classes)
        """
        return self.mlp(x)


class DeceptionNet(nn.Module):
    """
    Modello completo per Deception Detection.
    Pipeline: Video frames → ResNet → TCN → Attention → MLP → Prediction
    """

    def __init__(self, config):
        """
        Args:
            config: Oggetto Config con tutti i parametri
        """
        super().__init__()

        # 1. ResNet feature extractor
        self.feature_extractor = ResNetFeatureExtractor(
            architecture=config.get('model.resnet.architecture'),
            pretrained=config.get('model.resnet.pretrained'),
            freeze=config.get('model.resnet.freeze_layers')
        )
        feature_dim = self.feature_extractor.feature_dim

        # 2. Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            input_dim=feature_dim,
            hidden_channels=config.get('model.tcn.hidden_channels'),
            kernel_size=config.get('model.tcn.kernel_size'),
            dropout=config.get('model.tcn.dropout')
        )
        tcn_output_dim = self.tcn.output_dim

        # 3. Temporal Attention
        self.attention = TemporalAttention(
            input_dim=tcn_output_dim,
            hidden_dim=config.get('model.attention.hidden_dim')
        )

        # 4. MLP Classifier
        self.classifier = MLPClassifier(
            input_dim=tcn_output_dim,
            hidden_dims=config.get('model.classifier.hidden_dims'),
            num_classes=config.get('model.classifier.num_classes'),
            dropout=config.get('model.classifier.dropout')
        )

        self.config = config

    def forward(self, frames, mask=None):
        """
        Forward pass completo.

        Args:
            frames: (batch, seq_len, C, H, W) - Video frames
            mask: (batch, seq_len) - True per frame reali, False per padding

        Returns:
            logits: (batch, num_classes) - Predizioni
            attention_weights: (batch, seq_len) - Pesi attention (per visualizzazione)
        """
        batch_size, seq_len, C, H, W = frames.shape

        # 1. Estrai feature con ResNet
        # Reshape: (batch, seq_len, C, H, W) → (batch*seq_len, C, H, W)
        frames_flat = frames.view(batch_size * seq_len, C, H, W)

        # Forward ResNet
        features = self.feature_extractor(frames_flat)  # (batch*seq_len, feature_dim)

        # Reshape back: (batch*seq_len, feature_dim) → (batch, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)

        # 2. Temporal modeling con TCN
        tcn_out = self.tcn(features, mask)  # (batch, seq_len, tcn_dim)

        # 3. Attention pooling
        context, attention_weights = self.attention(tcn_out, mask)  # (batch, tcn_dim)

        # 4. Classificazione finale
        logits = self.classifier(context)  # (batch, num_classes)

        return logits, attention_weights

    def predict(self, frames, mask=None):
        """
        Predizione con probabilità.

        Args:
            frames: (batch, seq_len, C, H, W)
            mask: (batch, seq_len)

        Returns:
            predictions: (batch,) - Classe predetta (0 o 1)
            probabilities: (batch, num_classes) - Probabilità per classe
        """
        logits, _ = self.forward(frames, mask)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities


def build_model(config):
    """
    Factory function per costruire il modello.

    Args:
        config: Oggetto Config

    Returns:
        model: DeceptionNet
        device: torch.device
    """
    model = DeceptionNet(config)

    device = torch.device(config['training']['device'])
    model = model.to(device)

    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 70}")
    print("MODELLO COSTRUITO")
    print(f"{'=' * 70}")
    print(f"Architettura: {config.get('model.resnet.architecture')} → TCN → Attention → MLP")
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainable: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'=' * 70}\n")

    return model, device


# ============ TEST DEL MODELLO ============

if __name__ == "__main__":
    """Test standalone del modello."""


    # Mock config per test
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
                        'hidden_channels': [256, 256, 256],
                        'kernel_size': 3,
                        'dropout': 0.2
                    },
                    'attention': {
                        'hidden_dim': 128
                    },
                    'classifier': {
                        'hidden_dims': [128, 64],
                        'num_classes': 2,
                        'dropout': 0.3
                    }
                },
                'training': {
                    'device': 'cpu'
                }
            }

        def get(self, key, default=None):
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value.get(k, {})
            return value if value != {} else default

        def __getitem__(self, key):
            return self._config[key]


    print("Test DeceptionNet")
    print("=" * 70)

    config = MockConfig()
    model, device = build_model(config)

    # Test forward pass
    batch_size = 2
    seq_len = 10
    frames = torch.randn(batch_size, seq_len, 3, 224, 224)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 8:] = False  # Simula padding su primo sample

    print("Input:")
    print(f"  Frames: {frames.shape}")
    print(f"  Mask: {mask.shape}")

    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(frames, mask)

    print("\nOutput:")
    print(f"  Logits: {logits.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    print(f"  Predictions: {torch.argmax(logits, dim=1)}")

    print("\n✓ Test completato!")