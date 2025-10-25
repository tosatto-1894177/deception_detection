import cv2
import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import frame_dataset


# ============ FACE DETECTION ============

class FaceDetector:

    def __init__(self, device):
        """
        Args:
            device: 'cpu' o 'cuda'
        """
        self.device = device
        self._init_haar()

    def _init_haar(self):
        """Inizializza Haar Cascade (fallback veloce)."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        print("✓ Haar Cascade caricato")

    def detect_and_crop(self, frame, margin=0.3, target_size=(224, 224)):
        """
        Rileva volto e ritorna crop centrato.

        Args:
            frame: numpy array (H, W, 3) in RGB
            margin: percentuale di margine attorno al volto (0.3 = 30%)
            target_size: dimensione output

        Returns:
            face_crop: numpy array del volto ritagliato, None se non trovato
        """
        return self._detect_haar(frame, margin, target_size)

    def _detect_haar(self, frame, margin, target_size):
        """Face detection con Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Prendi volto più grande
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]

        return self._crop_with_margin(frame, x, y, x + w, y + h, margin, target_size)

    @staticmethod
    def _crop_with_margin(frame, x1, y1, x2, y2, margin, target_size):
        """Helper per crop con margine."""
        h, w = frame.shape[:2]

        # Aggiungi margine
        box_w, box_h = x2 - x1, y2 - y1
        margin_w = int(box_w * margin)
        margin_h = int(box_h * margin)

        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)

        # Crop e resize
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        face_resized = cv2.resize(face_crop, target_size)
        return face_resized


# ============ ESTRAZIONE FRAME CON FACE CROP ============

def extract_frames_with_faces(video_path, output_dir, fps=5,
                              face_detector=None, img_size=(224, 224),
                              save_failed_frames=False):
    """
    Estrae frame con face detection e crop.

    Args:
        video_path: Path al video
        output_dir: Directory output
        fps: Frame per second da estrarre
        face_detector: Istanza di FaceDetector
        img_size: Dimensione finale
        save_failed_frames: Se True, salva frame senza volto come fallback

    Returns:
        dict con statistiche
    """
    os.makedirs(output_dir, exist_ok=True)

    if face_detector is None:
        face_detector = FaceDetector(device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    frame_interval = max(1, int(video_fps / fps))

    frame_count = 0
    saved_count = 0
    failed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Converti BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect e crop volto
            face_crop = face_detector.detect_and_crop(
                frame_rgb,
                margin=0.3,
                target_size=img_size
            )

            if face_crop is not None:
                # Salva face crop
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, face_bgr)
                saved_count += 1
            else:
                failed_count += 1
                if save_failed_frames:
                    # Salva frame completo come fallback
                    frame_resized = cv2.resize(frame, img_size)
                    frame_filename = f"frame_{saved_count:06d}_noface.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame_resized)
                    saved_count += 1

        frame_count += 1

    cap.release()

    detection_rate = (saved_count / (saved_count + failed_count) * 100) if (saved_count + failed_count) > 0 else 0

    return {
        'duration': duration,
        'saved': saved_count,
        'failed': failed_count,
        'detection_rate': detection_rate
    }


def process_dolos_with_faces(video_dir, output_base_dir, device,
                             fps=5, img_size=(224, 224)):
    """
    Processa dataset DOLOS con face detection.
    """
    video_dir = Path(video_dir)
    output_base_dir = Path(output_base_dir)

    # Inizializza face detector
    face_detector = FaceDetector(device=device)

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.rglob(ext))

    print(f"Trovati {len(video_files)} video\n")

    stats = {
        'success': 0,
        'failed': 0,
        'total_frames': 0,
        'total_failed_detections': 0
    }

    for video_path in tqdm(video_files, desc="Processing"):
        try:
            relative_path = video_path.relative_to(video_dir)
            output_dir = output_base_dir / relative_path.parent / video_path.stem

            result = extract_frames_with_faces(
                video_path=video_path,
                output_dir=output_dir,
                fps=fps,
                face_detector=face_detector,
                img_size=img_size,
                save_failed_frames=True  # Salva comunque frame senza volto
            )

            print(f"{video_path.name}: {result['saved']} frames, "
                  f"{result['detection_rate']:.1f}% detection rate")

            stats['success'] += 1
            stats['total_frames'] += result['saved']
            stats['total_failed_detections'] += result['failed']

        except Exception as e:
            print(f"\n⚠ Errore {video_path.name}: {str(e)}")
            stats['failed'] += 1

    print(f"\n{'=' * 50}")
    print(f"Completato!")
    print(f"Video processati: {stats['success']}/{len(video_files)}")
    print(f"Frame totali: {stats['total_frames']}")
    print(f"Face detection failures: {stats['total_failed_detections']}")
    print(
        f"Detection rate globale: {(stats['total_frames'] / (stats['total_frames'] + stats['total_failed_detections']) * 100):.1f}%")
    print(f"{'=' * 50}")



def collate_fn_with_padding(batch):
    """Collate con padding per sequenze variabili."""
    frames_list, labels, lengths = zip(*batch)

    max_len = max(lengths)
    batch_size = len(frames_list)
    C, H, W = frames_list[0].shape[1:]

    padded_frames = torch.zeros(batch_size, max_len, C, H, W)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (frames, length) in enumerate(zip(frames_list, lengths)):
        padded_frames[i, :length] = frames
        mask[i, :length] = 1

    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_frames, labels, lengths, mask