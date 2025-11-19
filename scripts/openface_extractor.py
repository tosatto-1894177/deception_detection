"""
OpenFace Feature Extractor
Estrae Action Units, gaze, pose e landmarks da tutte le clip DOLOS

Modificato per estrazione da clip invece che da frame

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
        DEPRECATO
        Estrae feature da directory di frame
        In seguito ad analisi si è evidenziata una bassa qualità delle feature

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
        DEPRECATO
        Processa tutte i frame di una clip in una cartella

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


    def extract_from_video(self, video_path, output_name, fps=5):
        """
        Estrae feature direttamente da un file video

        Args:
            video_path: Path al file video (.mp4)
            output_name: Nome file output CSV (senza estensione)
            fps: Frame per secondo da campionare (default=5, coerente con preprocessing)

        Returns:
            Path al CSV generato o None se fallito
        """
        video_path = Path(video_path).resolve()
        output_csv = self.output_dir / f"{output_name}.csv"

        # Skip se già esiste
        if output_csv.exists():
            return output_csv

        # Verifica che il video esista
        if not video_path.exists():
            print(f"❌ Video non trovato: {video_path}")
            return None

        # Crea temp dir per output intermedio
        temp_dir = (self.output_dir / "temp").resolve()
        temp_dir.mkdir(exist_ok=True)

        try:
            # Comando OpenFace ottimizzato per estrazione da video
            cmd = [
                str(self.openface_path),
                "-f", str(video_path),  # Input: file video
                "-out_dir", str(temp_dir),  # Directory output temporanea
                "-fps", str(fps),  # Campiona a fps specificati (default=5)
                "-aus",  # Estrae Action Units (regression + classification)
                "-pose",  # Estrae head pose (rotation + translation)
                "-gaze",  # Estrae gaze direction
                "-2Dfp",  # Estrae 2D facial landmarks (68 punti)
                "-3Dfp",  # Estrae 3D facial landmarks (più robusto)
                "-pdmparams",  # Parametri Point Distribution Model
                "-tracked",  # Abilita tracking temporale tra frame
            ]

            # Esegue OpenFace
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            # Trova CSV generato da OpenFace
            generated_csvs = list(temp_dir.glob("*.csv"))

            if not generated_csvs:
                print(f"❌ OpenFace fallito per: {video_path.name}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}")
                return None

            # Verifica qualità estrazione
            df = pd.read_csv(generated_csvs[0])

            # Conta frame processati con successo
            if 'success' in df.columns:
                success_rate = df['success'].mean()
                if success_rate < 0.5:  # Se meno del 50% frame hanno successo
                    print(f"⚠️  Bassa success rate per {video_path.name}: {success_rate:.1%}")

            # Conta frame con confidence bassa
            if 'confidence' in df.columns:
                low_confidence = (df['confidence'] < 0.7).sum()
                if low_confidence > len(df) * 0.3:  # Se più del 30% ha bassa confidence
                    print(f"⚠️  Molti frame con bassa confidence per {video_path.name}: {low_confidence}/{len(df)}")

            # Sposta CSV in directory finale con nome corretto
            shutil.move(str(generated_csvs[0]), str(output_csv))

            return output_csv

        except subprocess.CalledProcessError as e:
            print(f"❌ Errore OpenFace per {video_path.name}:")
            print(f"   {e.stderr}")
            return None

        except Exception as e:
            print(f"❌ Errore generico per {video_path.name}: {e}")
            return None

        finally:
            # Pulisci temp dir
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    def process_all_videos(self, video_dir, fps=5):
        """
        Processa tutti i video in una cartella

        Args:
            video_dir: Cartella contenente i file .mp4
            fps: Frame per secondo da campionare (default=5)

        Returns:
            dict: Statistiche sull'estrazione
        """
        video_dir = Path(video_dir)

        print("\n" + "=" * 80)
        print("ESTRAZIONE FEATURE OPENFACE DA VIDEO")
        print("=" * 80)
        print(f"Cartella video: {video_dir}")
        print(f"Output CSV: {self.output_dir}")
        print(f"FPS campionamento: {fps}")
        print("=" * 80 + "\n")

        # Trova tutti i video .mp4
        video_files = list(video_dir.glob("*.mp4"))

        if not video_files:
            print(f"❌ Nessun file .mp4 trovato in {video_dir}")
            return {'successful': 0, 'failed': 0, 'skipped': 0, 'total': 0}

        print(f"Trovati {len(video_files)} video da processare\n")

        successful = 0
        failed = 0
        skipped = 0

        # Processa ogni video
        for video_path in tqdm(video_files, desc="Processing videos"):
            # Nome clip = nome file senza estensione
            clip_name = video_path.stem
            output_csv = self.output_dir / f"{clip_name}.csv"

            # Skip se già processato
            if output_csv.exists():
                skipped += 1
                continue

            # Estrai features
            result = self.extract_from_video(video_path, clip_name, fps=fps)

            if result:
                successful += 1
            else:
                failed += 1

        # Statistiche finali
        print("\n" + "=" * 80)
        print("ESTRAZIONE COMPLETATA")
        print("=" * 80)
        print(f"✅ Successo: {successful}")
        print(f"   Già processati: {skipped}")
        print(f"❌ Falliti: {failed}")
        print(f"   Totale: {successful + skipped + failed}")
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

        Returns:
            bool: True se valido, False altrimenti
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
            au_intensity = [col for col in df.columns if col.startswith(' AU') and '_r' in col]
            au_presence = [col for col in df.columns if col.startswith(' AU') and '_c' in col]
            gaze = [col for col in df.columns if 'gaze' in col.lower()]
            pose = [col for col in df.columns if 'pose' in col.lower()]
            landmarks = [col for col in df.columns if col.startswith(' x_') or col.startswith(' y_')]

            print(f"\nFeature breakdown:")
            print(f"  Action Units (intensity): {len(au_intensity)}")
            print(f"  Action Units (presence):  {len(au_presence)}")
            print(f"  Gaze:                     {len(gaze)}")
            print(f"  Pose:                     {len(pose)}")
            print(f"  Landmarks:                {len(landmarks)}")

            # Verifica qualità
            if 'success' in df.columns:
                success_rate = df['success'].mean()
                print(f"\nQualità estrazione:")
                print(f"  Success rate: {success_rate:.1%}")
                print(f"  Frame con successo: {df['success'].sum()}/{len(df)}")

            if 'confidence' in df.columns:
                avg_conf = df['confidence'].mean()
                low_conf = (df['confidence'] < 0.7).sum()
                print(f"  Confidence media: {avg_conf:.3f}")
                print(f"  Frame con bassa confidence (<0.7): {low_conf}/{len(df)}")

            # Sample features
            print(f"\nSample AU intensities:")
            for au in au_intensity[:5]:
                mean_val = df[au].mean()
                std_val = df[au].std()
                print(f"  {au}: {mean_val:.3f} ± {std_val:.3f}", end="")

                # Warning se costante
                if std_val < 0.01:
                    print(" ⚠️  (quasi costante!)")
                else:
                    print(" ✅")

            print(f"{'=' * 80}\n")

            return True

        except Exception as e:
            print(f"❌ Errore leggendo CSV: {e}")
            return False


def main():
    """Script principale per estrazione"""

    # Path all'eseguibile OpenFace per estrarre le feature
    OPENFACE_PATH = "C:/Users/lucat/OpenFace/FeatureExtraction.exe"

    # Cartella con i video (.mp4) DOLOS
    VIDEO_DIR = Path("data/raw_clips").resolve()

    # Cartella output per CSV OpenFace
    OUTPUT_DIR = Path("data/openface2").resolve()

    # FPS per campionamento (coerente con preprocessing video)
    FPS = 5

    print(f"\n📂 Configurazione paths:")
    print(f"   Cartella clip: {VIDEO_DIR}")
    print(f"   Cartella output: {OUTPUT_DIR}")
    print(f"   FPS campionamento: {FPS}")
    print(f"   OpenFace: {OPENFACE_PATH}")

    # Verifica che la directory video esista
    if not VIDEO_DIR.exists():
        print(f"\n❌ ERRORE: Directory video non trovata: {VIDEO_DIR}")
        print("   Assicurati che i video siano in data/raw_clips/")
        return

    # Crea extractor
    extractor = OpenFaceExtractor(
        openface_path=OPENFACE_PATH,
        output_dir=OUTPUT_DIR
    )

    # Processa tutte le clip
    stats = extractor.process_all_videos(video_dir=VIDEO_DIR, fps=FPS)

    if stats['successful'] > 0:
        print("\n🔍 Verifica qualità CSV generati (primi 3):")
        csv_files = sorted(Path(OUTPUT_DIR).glob("*.csv"))[:3]
        for csv_file in csv_files:
            extractor.verify_csv(csv_file)

    # Summary finale
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Video processati con successo: {stats['successful']}")
    print(f"Video già processati (skip): {stats['skipped']}")
    print(f"Video falliti: {stats['failed']}")
    print(f"Total video: {stats['total']}")

    if stats['successful'] > 0:
        print(f"\n✅ Estrazione completata!")
        print(f"   CSV salvati in: {OUTPUT_DIR}")
        print(f"   Prossimo step: Aggiorna config.yaml con:")
        print(f"   paths:")
        print(f"     openface_csv_dir: 'data/openface2'")
    elif stats['failed'] > 0:
        print(f"\n⚠️  Alcuni video sono falliti. Controlla i log sopra.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()