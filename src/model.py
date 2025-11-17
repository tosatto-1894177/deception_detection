"""
DcDtModel: Modello completo per Deception Detection
Architettura: ResNet → TCN → Attention → MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights


class ResNetFeatureExtractor(nn.Module):
    """Wrapper per ResNet come estrattore di feature"""

    def __init__(self, architecture='resnet34', pretrained=True, freeze=True):
        super().__init__()

        # Carica ResNet pre-addestrata
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

        # Prima convoluzione
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        # Seconda convoluzione
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
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
        residual = x # Salva input per skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual) # Adatta dimensioni

        out += residual
        out = F.relu(out)
        out = self.dropout2(out)

        return out


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network per catturare dipendenze temporali"""

    def __init__(self, input_dim, hidden_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            input_dim: Dimensione feature input (512 per ResNet34)
            hidden_channels: Lista con numero canali per ogni layer [256, 256, 256]
            kernel_size: Dimensione kernel convoluzionale (3 -> guarda 3 frame alla volta)
            dropout: Dropout rate (percentuale di neuroni "spenti")
        """
        super().__init__()

        layers = []

        for i in range(len(hidden_channels)):
            in_ch = input_dim if i == 0 else hidden_channels[i - 1]
            out_ch = hidden_channels[i]

            # Crea blocco convoluzionale temporale
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

        out = self.network(x) # Passa per 3 TemporalConvBlock

        # Torna a (batch, seq_len, channels)
        out = out.transpose(1, 2)

        # Applica mask per azzerare padding
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class TemporalAttention(nn.Module):
    """Attention mechanism per aggregare frame temporali"""

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
        # Calcola attention scores per ogni frame
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Maschera frame di padding
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Normalizza con softmax
        attention_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), x)  # (batch, 1, feature_dim)
        context = context.squeeze(1)  # (batch, feature_dim)

        return context, attention_weights


class MLPClassifier(nn.Module):
    """MLP finale per classificazione"""

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

        # Crea hidden layer
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


class DcDtModel(nn.Module):
    """
    Modello completo per Deception Detection
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
        Forward pass completo

        Args:
            frames: (batch, seq_len, C, H, W) = (8, 50, 3, 224, 224)
            mask: (batch, seq_len) = (8, 50) - True per frame reali, False per padding

        Returns:
            logits: (batch, num_classes) = (8, 2) - Predizioni
            attention_weights: (batch, seq_len) = (8, 50) - Pesi attention (per visualizzazione)
        """
        batch_size, seq_len, C, H, W = frames.shape

        # 1. Estrai feature con ResNet
        # Reshape: (batch, seq_len, C, H, W) → (batch*seq_len, C, H, W) = (8, 50, 3, 224, 224) → (400, 3, 224, 224)
        frames_flat = frames.view(batch_size * seq_len, C, H, W)

        # Forward ResNet = (400, 3, 224, 224) → (400, 512) = (batch*seq_len, feature_dim)
        features = self.feature_extractor(frames_flat)

        # Reshape back: (batch*seq_len, feature_dim) → (batch, seq_len, feature_dim) = (400, 512) → (8, 50, 512)
        features = features.view(batch_size, seq_len, -1)

        # 2. Cattura dipendenze temporali con TCN -> (batch, seq_len, tcn_dim) = (8, 50, 512) → (8, 50, 256)
        tcn_out = self.tcn(features, mask)

        # 3. Attention pooling - aggrega frame pesati per importanza
        # (8, 50, 256) → (8, 256) + (8, 50) = (batch, tcn_dim)
        context, attention_weights = self.attention(tcn_out, mask)

        # 4. Classificazione finale - score per Truth/Deception
        # (8, 256) → (8, 2) = (batch, num_classes)
        logits = self.classifier(context)

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


class OpenFaceBranch(nn.Module):
    """
    Branch dedicato per processare le features OpenFace features

    Architettura:
    Input (B, T, 49) → MLP Encoder → TCN → Attention → Context (B, 128)
    """

    def __init__(self, input_dim=49, hidden_dim=128, tcn_channels=None,
                 kernel_size=3, dropout=0.2):
        """
        Args:
            input_dim: Dimensione input OpenFace (default 49)
            hidden_dim: Dimensione hidden layer (default 128)
            tcn_channels: Lista canali TCN (default [128])
            kernel_size: Kernel size per TCN
            dropout: Dropout rate
        """
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [hidden_dim]

        # 1. MLP Encoder: normalizza e proietta le features OpenFace
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. TCN: cattura dinamiche temporali
        self.tcn = TemporalConvNet(
            input_dim=hidden_dim,
            hidden_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 3. Attention: aggrega frame temporali
        self.attention = TemporalAttention(
            input_dim=tcn_channels[-1],
            hidden_dim=64
        )

        self.output_dim = tcn_channels[-1]

    def forward(self, openface_features, mask=None):
        """
        Forward pass del branch OpenFace

        Args:
            openface_features: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - True per frame reali

        Returns:
            context: (batch, output_dim) - Context vector aggregato
            attention_weights: (batch, seq_len) - Pesi attention
        """
        # 1. Encode features con MLP: (B, T, 49) → (B, T, hidden_dim)
        x = self.encoder(openface_features)

        # 2. Modeling temporale con TCN: (B, T, hidden_dim) → (B, T, output_dim)
        x = self.tcn(x, mask)

        # 3. Aggrega con Attention: (B, T, output_dim) → (B, output_dim)
        context, attention_weights = self.attention(x, mask)

        return context, attention_weights


class DcDtModelV2(nn.Module):
    """
    Modello Multimodale con Late Fusion

    Combina due modalità:
    1. VIDEO: Frames → ResNet → TCN → Attention → (B, 256)
    2. OPENFACE: AUs/Gaze/Pose → MLP → TCN → Attention → (B, 128)

    Late Fusion: Concatena i context vectors → MLP → Prediction
    """

    def __init__(self, config):
        """
        Args:
            config: config.yaml
        """
        super().__init__()

        self.config = config

        # Branch video
        self.video_feature_extractor = ResNetFeatureExtractor(
            architecture=config.get('model.resnet.architecture'),
            pretrained=config.get('model.resnet.pretrained'),
            freeze=config.get('model.resnet.freeze_layers')
        )
        video_feature_dim = self.video_feature_extractor.feature_dim

        self.video_tcn = TemporalConvNet(
            input_dim=video_feature_dim,
            hidden_channels=config.get('model.tcn.hidden_channels'),
            kernel_size=config.get('model.tcn.kernel_size'),
            dropout=config.get('model.tcn.dropout')
        )
        video_tcn_output_dim = self.video_tcn.output_dim

        self.video_attention = TemporalAttention(
            input_dim=video_tcn_output_dim,
            hidden_dim=config.get('model.attention.hidden_dim')
        )

        # Branch Openface
        openface_config = config.get('model.openface', {})
        openface_input_dim = openface_config.get('input_dim', 49)
        openface_hidden_dim = openface_config.get('hidden_dim', 128)
        openface_tcn_channels = openface_config.get('tcn_channels', [128])

        self.openface_branch = OpenFaceBranch(
            input_dim=openface_input_dim,
            hidden_dim=openface_hidden_dim,
            tcn_channels=openface_tcn_channels,
            kernel_size=config.get('model.tcn.kernel_size', 3),
            dropout=config.get('model.tcn.dropout', 0.2)
        )

        # Fusion layer -> Late fusion
        fusion_input_dim = video_tcn_output_dim + self.openface_branch.output_dim
        fusion_hidden_dims = config.get('model.fusion.hidden_dims', [256, 128])

        # Classificatore finale
        self.fusion_classifier = MLPClassifier(
            input_dim=fusion_input_dim,
            hidden_dims=fusion_hidden_dims,
            num_classes=config.get('model.classifier.num_classes', 2),
            dropout=config.get('model.classifier.dropout', 0.3)
        )

        print(f"\n{'=' * 80}")
        print("LATE FUSION MODEL")
        print(f"{'=' * 80}")
        print(f"Video context dim:    {video_tcn_output_dim}")
        print(f"OpenFace context dim: {self.openface_branch.output_dim}")
        print(f"Fused dim:            {fusion_input_dim}")
        print(f"{'=' * 80}\n")

    def forward(self, frames, mask=None, openface_features=None):
        """
        Forward pass multimodale

        Args:
            frames: (batch, seq_len, C, H, W) - Video frames
            mask: (batch, seq_len) - Mask per padding
            openface_features: (batch, seq_len, 49) - OpenFace features (opzionale)

        Returns:
            logits: (batch, num_classes) - Predizioni
            attention_dict: dict con attention weights da entrambi i branch
        """
        batch_size, seq_len, C, H, W = frames.shape

        # Branch video

        # 1. Estrae feature visive con ResNet
        frames_flat = frames.view(batch_size * seq_len, C, H, W)
        video_features = self.video_feature_extractor(frames_flat)
        video_features = video_features.view(batch_size, seq_len, -1)

        # 2. Modeling temporale con TCN
        video_tcn_out = self.video_tcn(video_features, mask)

        # 3. Aggrega con Attention
        video_context, video_attention = self.video_attention(video_tcn_out, mask)

        # Branch Openface
        if openface_features is not None:
            # Processa features OpenFace
            openface_context, openface_attention = self.openface_branch(
                openface_features, mask
            )
        else:
            # Fallback: se OpenFace manca, usa zeros
            # (permette di gestire clip senza OpenFace features)
            openface_context = torch.zeros(
                batch_size,
                self.openface_branch.output_dim,
                device=frames.device
            )
            openface_attention = torch.zeros(batch_size, seq_len, device=frames.device)

        # ===== LATE FUSION =====

        # Concatena context vectors: (B, 256) + (B, 128) → (B, 384)
        fused_context = torch.cat([video_context, openface_context], dim=1)

        # Classificazione finale
        logits = self.fusion_classifier(fused_context)

        # Ritorna attention weights per entrambi i branch (utile per analisi)
        attention_dict = {
            'video': video_attention,
            'openface': openface_attention
        }

        return logits, attention_dict

    def predict(self, frames, mask=None, openface_features=None):
        """
        Predizione con probabilità

        Args:
            frames: (batch, seq_len, C, H, W)
            mask: (batch, seq_len)
            openface_features: (batch, seq_len, 49)

        Returns:
            predictions: (batch,) - Classe predetta
            probabilities: (batch, num_classes) - Probabilità
        """
        logits, _ = self.forward(frames, mask, openface_features)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities


def build_model(config):
    """
    Factory function per costruire il modello

    Args:
        config: config.yaml

    Returns:
        model: DcDtModel o DcDtModelV2 (con late fusion)
        device: torch.device
    """
    model_type = config.get('model.type', 'video_only')

    if model_type == 'multimodal':
        print("\nCostruendo modello multimodale (video + OpenFace)")
        model = DcDtModelV2(config)
    elif model_type == 'video_only':
        print("\nCostruendo modello VIDEO-ONLY")
        model = DcDtModel(config)
    else:
        raise ValueError(f"Model type '{model_type}' non supportato. "
                         f"Usa 'video_only' o 'multimodal'")

    device = torch.device(config['training']['device'])
    model = model.to(device)

    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 80}")
    print("MODELLO COSTRUITO")
    print(f"{'=' * 80}")
    print(f"Tipo: {model_type.upper()}")
    print(f"Architettura: {config.get('model.resnet.architecture')} → TCN → Attention → MLP")
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri trainable: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'=' * 80}\n")

    return model, device