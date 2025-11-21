"""
Script per analizzare distribuzione frame nel dataset preprocessato.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

def main():
    # Path ai frames
    frames_dir = Path("data/frames")

    # Analizza distribuzione
    frame_counts = []
    clip_names = []

    for clip_dir in frames_dir.rglob("*"):
        if clip_dir.is_dir():
            frames = list(clip_dir.glob("frame_*.jpg"))
            if len(frames) > 0:
                frame_counts.append(len(frames))
                clip_names.append(clip_dir.name)

    frame_counts = np.array(frame_counts)

    print("="*70)
    print("ANALISI DISTRIBUZIONE FRAME")
    print("="*70)
    print(f"\nClip totali: {len(frame_counts)}")
    print(f"\nStatistiche frame per clip:")
    print(f"  Min:     {frame_counts.min()}")
    print(f"  Max:     {frame_counts.max()}")
    print(f"  Media:   {frame_counts.mean():.1f}")
    print(f"  Mediana: {np.median(frame_counts):.1f}")
    print(f"  Std:     {frame_counts.std():.1f}")

    # Distribuzione per range di frame
    print(f"\nDistribuzione:")
    print(f"  0-30 frames:     {(frame_counts <= 30).sum()} clip ({(frame_counts <= 30).sum()/len(frame_counts)*100:.1f}%)")
    print(f"  31-50 frames:    {((frame_counts > 30) & (frame_counts <= 50)).sum()} clip ({((frame_counts > 30) & (frame_counts <= 50)).sum()/len(frame_counts)*100:.1f}%)")
    print(f"  51-100 frames:   {((frame_counts > 50) & (frame_counts <= 100)).sum()} clip ({((frame_counts > 50) & (frame_counts <= 100)).sum()/len(frame_counts)*100:.1f}%)")
    print(f"  101-500 frames:  {((frame_counts > 100) & (frame_counts <= 500)).sum()} clip ({((frame_counts > 100) & (frame_counts <= 500)).sum()/len(frame_counts)*100:.1f}%)")
    print(f"  500+ frames:     {(frame_counts > 500).sum()} clip ({(frame_counts > 500).sum()/len(frame_counts)*100:.1f}%)")

    # Clip più lunghe
    print(f"\nTop 10 clip più lunghe:")
    top_indices = np.argsort(frame_counts)[-10:]
    for idx in reversed(top_indices):
        print(f"  {clip_names[idx]}: {frame_counts[idx]} frames")

    # Plot distribuzione
    plt.figure(figsize=(12, 6))

    # --- GRAFICO 1: Scala Logaritmica con etichette reali ---
    plt.subplot(1, 2, 1)
    # log=True attiva la scala logaritmica per visualizzare le barre basse
    plt.hist(frame_counts[frame_counts <= 1000], bins=50, edgecolor='black', log=True)

    plt.xlabel('Numero frame')
    plt.ylabel('Numero clip')
    plt.title('Distribuzione Frame per Clip in scala logaritmica')

    # --- QUI LA MAGIA: Imposta i numeri reali (10, 100, 1000) ---
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())  # Usa numeri interi, non 10^x
    ax.ticklabel_format(axis='y', style='plain')  # Disabilita notazione scientifica

    plt.axvline(x=30, color='r', linestyle='--', label='max_frames=30')
    plt.axvline(x=50, color='g', linestyle='--', label='max_frames=50')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)  # 'which="both"' mostra la griglia anche per i valori intermedi

    # --- GRAFICO 2: Zoom Lineare (Invariato) ---
    plt.subplot(1, 2, 2)
    plt.hist(frame_counts[frame_counts <= 200], bins=50, edgecolor='black')
    plt.xlabel('Numero frame')
    plt.ylabel('Numero clip')
    plt.title('Distribuzione Frame per Clip (zoom 0-200)')
    plt.axvline(x=30, color='r', linestyle='--', label='max_frames=30')
    plt.axvline(x=50, color='g', linestyle='--', label='max_frames=50')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frame_distribution_log.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Grafico salvato in: frame_distribution_log.png")
    print("=" * 70)

if __name__ == "__main__":
    main()