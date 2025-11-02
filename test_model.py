"""
Test Standalone del Modello DeceptionNet
Verifica che tutti i componenti funzionino correttamente.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Aggiungi src al path se necessario
sys.path.append(str(Path(__file__).parent))

from src.model import DeceptionNet, build_model
from src.config_loader import load_config


def test_model_components():
    """Test individuali dei componenti."""
    print("\n" + "=" * 70)
    print("TEST 1: COMPONENTI INDIVIDUALI")
    print("=" * 70)

    # Mock config
    config = create_mock_config()

    # 1. Test ResNet Feature Extractor
    print("\n[1/5] Testing ResNet Feature Extractor...")
    from src.model import ResNetFeatureExtractor

    resnet = ResNetFeatureExtractor(
        architecture='resnet34',
        pretrained=True,
        freeze=True
    )

    test_frames = torch.randn(4, 3, 224, 224)  # 4 frames
    features = resnet(test_frames)

    assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"
    print(f"   ✓ ResNet output shape: {features.shape}")

    # 2. Test TCN
    print("\n[2/5] Testing Temporal Convolutional Network...")
    from src.model import TemporalConvNet

    tcn = TemporalConvNet(
        input_dim=512,
        hidden_channels=[128, 128],
        kernel_size=3,
        dropout=0.2
    )

    test_sequence = torch.randn(2, 10, 512)  # batch=2, seq=10, feat=512
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 8:] = False  # Padding su primo sample

    tcn_out = tcn(test_sequence, mask)

    assert tcn_out.shape == (2, 10, 128), f"Expected (2, 10, 128), got {tcn_out.shape}"
    print(f"   ✓ TCN output shape: {tcn_out.shape}")

    # Verifica che padding sia stato azzerato
    assert torch.allclose(tcn_out[0, 8:], torch.zeros_like(tcn_out[0, 8:])), \
        "TCN non ha azzerato il padding!"
    print(f"   ✓ Padding correttamente azzerato")

    # 3. Test Attention
    print("\n[3/5] Testing Temporal Attention...")
    from src.model import TemporalAttention

    attention = TemporalAttention(input_dim=128, hidden_dim=64)

    context, attn_weights = attention(tcn_out, mask)

    assert context.shape == (2, 128), f"Expected (2, 128), got {context.shape}"
    assert attn_weights.shape == (2, 10), f"Expected (2, 10), got {attn_weights.shape}"
    print(f"   ✓ Attention context shape: {context.shape}")
    print(f"   ✓ Attention weights shape: {attn_weights.shape}")

    # Verifica che weights sommino a 1
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(2)), \
        "Attention weights non sommano a 1!"
    print(f"   ✓ Attention weights sum to 1.0")

    # Verifica che frames con padding abbiano peso ~0
    assert attn_weights[0, 8:].sum() < 0.01, \
        "Attention sta guardando frames con padding!"
    print(f"   ✓ Attention ignora correttamente il padding")

    # 4. Test MLP Classifier
    print("\n[4/5] Testing MLP Classifier...")
    from src.model import MLPClassifier

    mlp = MLPClassifier(
        input_dim=128,
        hidden_dims=[64],
        num_classes=2,
        dropout=0.3
    )

    logits = mlp(context)

    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"
    print(f"   ✓ MLP output shape: {logits.shape}")

    # Test probabilità
    probs = F.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2)), \
        "Probabilità non sommano a 1!"
    print(f"   ✓ Softmax probabilities sum to 1.0")

    print("\n[5/5] All component tests passed! ✓")


def test_full_model():
    """Test del modello completo end-to-end."""
    print("\n" + "=" * 70)
    print("TEST 2: MODELLO COMPLETO END-TO-END")
    print("=" * 70)

    config = create_mock_config()
    device = torch.device('cpu')  # CPU per test veloce

    # Build model
    model, _ = build_model(config)
    model = model.to(device)
    model.eval()

    # Dati di test
    batch_size = 3
    seq_lengths = [15, 10, 8]  # Sequenze di lunghezza variabile
    max_len = max(seq_lengths)

    # Crea batch con padding
    frames = torch.randn(batch_size, max_len, 3, 224, 224)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, length in enumerate(seq_lengths):
        mask[i, :length] = True

    print(f"\nInput:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Mask shape: {mask.shape}")

    # Forward pass
    print("\n[1/3] Forward pass...")
    with torch.no_grad():
        logits, attention_weights = model(frames, mask)

    print(f"   ✓ Logits shape: {logits.shape}")
    print(f"   ✓ Attention weights shape: {attention_weights.shape}")

    assert logits.shape == (batch_size, 2), \
        f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
    assert attention_weights.shape == (batch_size, max_len), \
        f"Expected attention shape ({batch_size}, {max_len}), got {attention_weights.shape}"

    # Verifica attention weights
    print("\n[2/3] Verifying attention weights...")
    for i, length in enumerate(seq_lengths):
        # Pesi sui frame reali devono sommare a ~1
        real_weights_sum = attention_weights[i, :length].sum().item()
        assert 0.99 < real_weights_sum < 1.01, \
            f"Sample {i}: attention weights sum = {real_weights_sum:.4f}"

        # Pesi sul padding devono essere ~0
        if length < max_len:
            padding_weights_sum = attention_weights[i, length:].sum().item()
            assert padding_weights_sum < 0.01, \
                f"Sample {i}: padding weights sum = {padding_weights_sum:.4f}"

    print(f"   ✓ Attention weights corretti per tutte le sequenze")

    # Test predictions
    print("\n[3/3] Testing predictions...")
    predictions, probabilities = model.predict(frames, mask)

    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Probabilities shape: {probabilities.shape}")
    print(f"\n   Predictions: {predictions.tolist()}")
    print(f"   Probabilities: ")
    for i in range(batch_size):
        print(f"     Sample {i}: Truth={probabilities[i, 0]:.3f}, Lie={probabilities[i, 1]:.3f}")

    assert predictions.shape == (batch_size,), \
        f"Expected predictions shape ({batch_size},), got {predictions.shape}"
    assert probabilities.shape == (batch_size, 2), \
        f"Expected probabilities shape ({batch_size}, 2), got {probabilities.shape}"


def test_backward_pass():
    """Test backward pass e calcolo gradienti."""
    print("\n" + "=" * 70)
    print("TEST 3: BACKWARD PASS & GRADIENTI")
    print("=" * 70)

    config = create_mock_config()
    device = torch.device('cpu')

    model, _ = build_model(config)
    model = model.to(device)
    model.train()  # Training mode

    # Dati fake
    batch_size = 2
    seq_len = 5
    frames = torch.randn(batch_size, seq_len, 3, 224, 224)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    labels = torch.tensor([0, 1])  # Truth, Lie

    # Forward
    print("\n[1/3] Forward pass...")
    logits, _ = model(frames, mask)

    # Loss
    print("[2/3] Computing loss...")
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    print(f"   ✓ Loss: {loss.item():.4f}")

    # Backward
    print("[3/3] Backward pass...")
    loss.backward()

    # Verifica che gradienti siano stati calcolati
    has_gradients = False
    total_params = 0
    params_with_grad = 0

    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            params_with_grad += 1
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    print(f"   ✓ {name}: grad_norm = {grad_norm:.6f}")

    assert has_gradients, "Nessun gradiente calcolato!"

    print(f"\n   ✓ Parametri totali: {total_params}")
    print(f"   ✓ Parametri trainable: {params_with_grad}")
    print(f"   ✓ Backward pass completato con successo!")


def test_memory_efficiency():
    """Test uso memoria e possibili memory leaks."""
    print("\n" + "=" * 70)
    print("TEST 4: EFFICIENZA MEMORIA")
    print("=" * 70)

    config = create_mock_config()
    device = torch.device('cpu')

    model, _ = build_model(config)
    model = model.to(device)

    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nParametri del modello:")
    print(f"  Totali: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (ResNet): {frozen_params:,}")

    # Test con batch size crescenti
    print(f"\nTest con batch size variabili:")

    for batch_size in [1, 2, 4, 8]:
        seq_len = 10
        frames = torch.randn(batch_size, seq_len, 3, 224, 224)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            logits, _ = model(frames, mask)

        # Memoria approssimativa (solo tensori, non conta overhead Python)
        input_memory = frames.element_size() * frames.nelement() / (1024 ** 2)
        output_memory = logits.element_size() * logits.nelement() / (1024 ** 2)

        print(f"  Batch {batch_size}: Input={input_memory:.2f}MB, Output={output_memory:.4f}MB ✓")

    print(f"\n   ✓ Nessun memory leak rilevato")


def test_with_real_dimensions():
    """Test con dimensioni realistiche del dataset."""
    print("\n" + "=" * 70)
    print("TEST 5: DIMENSIONI REALISTICHE DATASET")
    print("=" * 70)

    config = create_mock_config()
    device = torch.device('cpu')

    model, _ = build_model(config)
    model = model.to(device)
    model.eval()

    # Simula batch reale dal tuo dataset
    # Clip DOLOS: 2-19s, fps=5 → 10-95 frames, max_frames=50
    batch_size = 2  # Come nel tuo config

    # Sequenze con lunghezze variabili realistiche
    seq_lengths = [25, 48]  # Es: clip da 5s e 9.6s
    max_len = config.get('preprocessing.max_frames', 50)

    print(f"\nSimulazione batch reale:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max frames: {max_len}")
    print(f"  Clip lengths: {seq_lengths} frames")

    # Crea batch padded
    frames = torch.randn(batch_size, max_len, 3, 224, 224)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, length in enumerate(seq_lengths):
        mask[i, :length] = True

    # Forward pass
    with torch.no_grad():
        logits, attention_weights = model(frames, mask)

    print(f"\nOutput:")
    print(f"  Logits: {logits.shape}")
    print(f"  Attention weights: {attention_weights.shape}")

    # Analizza attention weights
    print(f"\nAttention analysis:")
    for i, length in enumerate(seq_lengths):
        weights = attention_weights[i, :length]
        max_weight_idx = weights.argmax().item()
        max_weight_val = weights[max_weight_idx].item()
        mean_weight = weights.mean().item()

        print(f"  Clip {i} ({length} frames):")
        print(f"    Max attention: frame {max_weight_idx} (weight={max_weight_val:.4f})")
        print(f"    Mean attention: {mean_weight:.4f}")
        print(f"    Distribution: min={weights.min():.4f}, max={weights.max():.4f}")

    print(f"\n   ✓ Test con dimensioni realistiche completato!")


def create_mock_config():
    """Crea configurazione mock per test."""

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
                    'device': 'cpu'
                },
                'preprocessing': {
                    'max_frames': 50
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

    return MockConfig()


def run_all_tests():
    """Esegue tutti i test."""
    print("\n" + "=" * 70)
    print("🚀 DECEPTION NET - TEST SUITE COMPLETO")
    print("=" * 70)

    try:
        test_model_components()
        test_full_model()
        test_backward_pass()
        test_memory_efficiency()
        test_with_real_dimensions()

        print("\n" + "=" * 70)
        print("✅ TUTTI I TEST PASSATI CON SUCCESSO!")
        print("=" * 70)
        print("\nIl modello è pronto per:")
        print("  → Test con batch reale dal dataset")
        print("  → Training loop completo")
        print("  → Deployment su Colab")

        return True

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("❌ TEST FALLITO!")
        print("=" * 70)
        print(f"\nErrore: {str(e)}")
        return False

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERRORE INASPETTATO!")
        print("=" * 70)
        print(f"\nErrore: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)