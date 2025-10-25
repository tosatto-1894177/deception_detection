import sys
import os
from multiprocessing.pool import ThreadPool

from yt_dlp import YoutubeDL
import ffmpeg

from pathlib import Path

# COMMAND = python YT_video_downloader.py videos/ csv_file_name.csv
# csv file format must be 'yt_video_id, file_name, start_time (in seconds), end_time (in seconds)'

class VidInfo:
    def __init__(self, yt_id, file_name, start_time, end_time, outdir):
        self.yt_id = yt_id
        self.start_time = start_time
        self.end_time = end_time
        self.out_filename = str(Path(outdir) / (file_name + '.mp4'))


def _parse_time_to_seconds(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    # accetta sia formati mm:ss che hh:mm:ss
    parts = s.split(':')
    if not all(p.isdigit() for p in parts):
        raise ValueError(f"Formato tempo non valido: {s}")
    parts = list(map(int, parts))
    if len(parts) == 2:      # mm:ss
        mm, ss = parts
        return mm*60 + ss
    elif len(parts) == 3:    # hh:mm:ss
        hh, mm, ss = parts
        return hh*3600 + mm*60 + ss
    else:
        raise ValueError(f"Formato tempo non valido: {s}")

def download(vidinfo):
    yt_base_url = 'https://www.youtube.com/watch?v='
    yt_url = yt_base_url + vidinfo.yt_id

    ydl_opts = {
        'format': '22/18',
        'quiet': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url=yt_url, download=False)
            download_url = info['url']
            http_headers = info.get('http_headers', {})
            total = info.get('duration')
    except:
        return_msg = '{}, ERROR (youtube)!'.format(vidinfo.yt_id)
        return return_msg

    start = float(vidinfo.start_time)
    end   = float(vidinfo.end_time)
    if total:
        try:
            end = min(end, float(total))
        except:
            pass
    duration = end - start
    if duration <= 0:
        return f"{vidinfo.yt_id}, ERROR (ffmpeg)!"

    # Utility: serializza gli header per ffmpeg (formato "Key: Value\r\n...")
    def _headers_to_str(h: dict) -> str:
        return ''.join([f"{k}: {v}\r\n" for k, v in h.items()])

    headers_arg = _headers_to_str(http_headers) if http_headers else None

    common_in_kwargs = {}
    if headers_arg:
        common_in_kwargs['headers'] = headers_arg
    # Utile per migliorare la robustezza di rete
    common_in_kwargs.update({
        'reconnect': 1,
        'reconnect_streamed': 1,
        'reconnect_on_network_error': 1,
        'rw_timeout': 15000000,
    })

    # Primo tentativo di download: -ss sull'INPUT, -t sull'OUTPUT
    try:
        inp = ffmpeg.input(download_url, ss=start, **common_in_kwargs)
        (
            inp
            .output(
                vidinfo.out_filename,
                t=duration, format='mp4',
                r=25, vcodec='libx264', crf=18, preset='veryfast',
                pix_fmt='yuv420p', acodec='aac', audio_bitrate=128000,
                strict='experimental'
            )
            .global_args('-y', '-loglevel', 'error')
            .run()
        )
        return f"{vidinfo.yt_id}, DONE!"
    except:
        pass

        # Secondo tentativo di download: sia -ss che -t sull'OUTPUT (seek accurato)
    try:
        inp = ffmpeg.input(download_url, **common_in_kwargs)
        (
            inp
            .output(
                vidinfo.out_filename,
                ss=start, t=duration,
                format='mp4',
                r=25, vcodec='libx264', crf=18, preset='veryfast',
                pix_fmt='yuv420p', acodec='aac', audio_bitrate=128000,
                strict='experimental'
            )
            .global_args('-y', '-loglevel', 'error')
            .run()
        )
        return f"{vidinfo.yt_id}, DONE!"
    except:
        return_msg = '{}, ERROR (ffmpeg)!'.format(vidinfo.yt_id)
        return return_msg


if __name__ == '__main__':

    split = sys.argv[1] # name of the destination folder
    csv_file = sys.argv[2]
    out_dir = split
    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(split)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        vidinfos = []
        for idx, line in enumerate(f):
            if not line.strip():
                continue  # salta righe vuote
            cols = [c.strip() for c in line.split(',')]
            # salta l’header (anche con BOM)
            if idx == 0 and cols[0].lower() in {'yt_video_id', 'yt_video', 'yt_id', 'video_id'}:
                continue
            # prende solo le prime 4 colonne (ignora eventuale 'label')
            yt_id, file_name, start_s, end_s = (cols + ["", "", "", ""])[:4]
            # converte i tempi in secondi (accetta numeri e mm:ss / hh:mm:ss)
            start_time = _parse_time_to_seconds(start_s)
            end_time = _parse_time_to_seconds(end_s)
            vidinfos.append(VidInfo(yt_id, file_name, start_time, end_time, out_dir))

    bad_files = (out_dir / "bad_files.txt").open("w", encoding="utf-8")
    results = ThreadPool(5).imap_unordered(download, vidinfos)
    cnt = 0
    for r in results:
        cnt += 1
        print(cnt, '/', len(vidinfos), r)
        if 'ERROR' in r:
            bad_files.write(r + '\n')
    bad_files.close()