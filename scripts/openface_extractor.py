"""
OpenFace Feature Extractor
Estrae Action Units, gaze, pose e landmarks da tutte le clip DOLOS

FEATURES ESTRATTE:
- Action Units (AUs): 17 intensity + 18 presence
- Gaze: direzione sguardo (4 features)
- Pose: rotazione testa (3 features)
- Landmarks: 68 punti facciali (136 features x,y)
"""

import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil


class OpenFaceExtractor:
    """
    Wrapper per OpenFace che processa video clips e salva CSV con features
    """

    def __init__(self, openface_path, output_dir):
        """
        Args:
            openface_path: path all'eseguibile FeatureExtraction di OpenFace
            output_dir: cartella dove vengono salvati i CSV
        """
        self.openface_path = Path(openface_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verifica che OpenFace sia installato
        if not self.openface_path.exists():
            raise FileNotFoundError(
                f"OpenFace non trovato in: {openface_path}\n"
            )

        print(f"✅ OpenFace trovato: {self.openface_path}")

    def extract_from_frames(self, frames_dir, output_name):
        """
        Estrae feature da directory di frame

        Args:
            frames_dir: Cartella con frame_*.jpg
            output_name: Nome file output CSV

        Returns:
            Path al CSV generato o None se fallito
        """
        frames_dir = Path(frames_dir).resolve()
        output_csv = self.output_dir / f"{output_name}.csv"

        if output_csv.exists():
            return output_csv

        # Crea temp dir
        temp_dir = (self.output_dir / "temp").resolve()
        temp_dir.mkdir(exist_ok=True)

        try:
            cmd = [ # Comando OpenFace
                str(self.openface_path),
                "-fdir", str(frames_dir),  # Directory frame
                "-out_dir", str(temp_dir), # directory output
                "-aus", # estrae Action Units
                "-pose", # estrae pose
                "-gaze", # estrae gaze
                "-2Dfp", # estrae 2D facial landmarks
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            # Trova CSV generato
            generated_csvs = list(temp_dir.glob("*.csv"))

            if not generated_csvs:
                print(f"❌ OpenFace fallito per: {frames_dir.name}")
                print(f"   Frames disponibili: {len(list(frames_dir.glob('frame_*.jpg')))}")
                return None

            # Sposta in directory finale
            shutil.move(str(generated_csvs[0]), str(output_csv))

            return output_csv

        except subprocess.CalledProcessError as e:
            print(f"❌ Errore OpenFace per {frames_dir.name}:")
            print(f"   {e.stderr}")
            return None

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def process_all_clips(self, frames_dir):
        """
        Processa tutte le clip in una cartella

        Args:
            frames_dir: Cartella contenente le sottocartelle di frame

        Returns:
            dict: Statistiche sull'estrazione
        """
        frames_dir = Path(frames_dir)

        print("\n" + "=" * 80)
        print("ESTRAZIONE FEATURE OPENFACE")
        print("=" * 80)
        print(f"Cartella frame: {frames_dir}")
        print(f"Output: {self.output_dir}")
        print("=" * 80 + "\n")

        clip_dirs = [d for d in frames_dir.rglob("*") if d.is_dir()]
        clip_dirs = [d for d in clip_dirs if list(d.glob("frame_*.jpg"))]

        print(f"Trovate {len(clip_dirs)} clip con frame\n")

        successful = 0
        failed = 0
        skipped = 0

        for clip_dir in tqdm(clip_dirs, desc="Processing clips"):
            clip_name = clip_dir.name
            output_csv = self.output_dir / f"{clip_name}.csv"

            if output_csv.exists():
                skipped += 1
                continue

            result = self.extract_from_frames(clip_dir, clip_name)

            if result:
                successful += 1
            else:
                failed += 1

        # Statistiche finali
        print("\n" + "=" * 80)
        print("ESTRAZIONE COMPLETATA")
        print("=" * 70)
        print(f"✅ Successo: {successful}")
        print(f"⏭️  Gia processati:   {skipped}")
        print(f"❌ Falliti: {failed}")
        print(f"📊 Totale: {successful + failed}")
        print(f"📂 CSV salvati in: {self.output_dir.resolve()}")
        print("=" * 80 + "\n")

        return {
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'total': successful + skipped + failed
        }

    def verify_csv(self, csv_path):
        """
        Verifica che un CSV OpenFace sia valido e mostra info

        Args:
            csv_path: Path al CSV
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            print(f"❌ CSV non trovato: {csv_path}")
            return False

        try:
            df = pd.read_csv(csv_path)

            print(f"\n{'=' * 80}")
            print(f"CSV: {csv_path.name}")
            print(f"{'=' * 80}")
            print(f"Frames: {len(df)}")
            print(f"Features: {len(df.columns)}")

            # Identifica colonne per tipo
            au_intensity = [col for col in df.columns if col.startswith('AU') and '_r' in col]
            au_presence = [col for col in df.columns if col.startswith('AU') and '_c' in col]
            gaze = [col for col in df.columns if 'gaze' in col.lower()]
            pose = [col for col in df.columns if 'pose' in col.lower()]
            landmarks = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]

            print(f"\nFeature breakdown:")
            print(f"  Action Units (intensity): {len(au_intensity)}")
            print(f"  Action Units (presence):  {len(au_presence)}")
            print(f"  Gaze:                     {len(gaze)}")
            print(f"  Pose:                     {len(pose)}")
            print(f"  Landmarks:                {len(landmarks)}")

            # Sample features
            print(f"\nSample AU intensities:")
            for au in au_intensity[:5]:
                print(f"  {au}: {df[au].mean():.3f} ± {df[au].std():.3f}")

            print(f"{'=' * 80}\n")

            return True

        except Exception as e:
            print(f"❌ Errore leggendo CSV: {e}")
            return False


def main():
    """Script principale per estrazione"""

    # Path all'eseguibile OpenFace per estrarre le feature
    OPENFACE_PATH = "C:/Users/lucat/OpenFace/FeatureExtraction.exe"

    # Directory con i frame estratti
    FRAMES_DIR = Path("data/frames").resolve()

    # Directory output per CSV OpenFace
    OUTPUT_DIR = Path("data/openface").resolve()

    # ====================================

    print(f"\n📂 Paths configurati:")
    print(f"   Frames: {FRAMES_DIR}")
    print(f"   Output: {OUTPUT_DIR}")

    # Crea extractor
    extractor = OpenFaceExtractor(
        openface_path=OPENFACE_PATH,
        output_dir=OUTPUT_DIR
    )

    # Processa tutte le clip
    stats = extractor.process_all_clips(frames_dir=FRAMES_DIR)

    # Verifica un CSV di esempio
    csv_files = list(Path(OUTPUT_DIR).glob("*.csv"))
    if csv_files:
        print("\n🔍 Verifica CSV di esempio:")
        extractor.verify_csv(csv_files[0])


if __name__ == "__main__":
    main()