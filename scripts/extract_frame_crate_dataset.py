# 1. Estrai frame con face detection
from frame_extractor import process_dolos_with_faces
from src.frame_dataset import DOLOSFrameDataset, collate_fn_with_padding
from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

process_dolos_with_faces(
    video_dir="path/to/video/clips",
    output_base_dir="path/to/face_frames",
    fps=5,
    device=device
)

dataset = DOLOSFrameDataset(
    root_dir="path/to/face_frames",
    annotation_file="train_dolos_example.csv",
    max_frames=100
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn_with_padding,
    num_workers=2
)

# 3. Testa
for frames, labels, lengths, mask, metadata in dataloader:
    print(f"Frames: {frames.shape}")
    print(f"Labels (0=truth, 1=lie): {labels}")
    print(f"Clip names: {metadata['clip_names']}")
    break