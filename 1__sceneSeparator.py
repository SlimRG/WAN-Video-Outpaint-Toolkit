import os
import subprocess
import glob
from pathlib import Path

import cv2
from transnetv2pt import predict_video

"""
Scene splitter using TransNetV2 and FFmpeg (NVENC) / Разделение видео на сцены с TransNetV2 и FFmpeg (NVENC)

EN:
- Detects scene boundaries with TransNetV2 (PyTorch port) and exports each scene as a separate MP4.
- Re-encodes with NVIDIA H.264 (NVENC), scales to 720p height, and writes to OUTPUT_DIR.
- Note: using "-vf scale=..." and "format=yuv420p" makes the output NOT truly lossless,
  even with "-tune lossless" (chroma subsampling + scaling introduce loss).

RU:
- Находит границы сцен TransNetV2 (PyTorch) и сохраняет каждую сцену отдельным MP4.
- Перекодирует через NVIDIA H.264 (NVENC), масштабирует по высоте до 720p и пишет в OUTPUT_DIR.
- Важно: при "-vf scale=..." и "format=yuv420p" файл НЕ будет полностью без потерь,
  даже с "-tune lossless" (масштаб и 4:2:0 дают потери).
"""

# ===
# = Paths / Пути
# ===

INPUT_DIR = Path(r"D:/Experiments/1__original")
OUTPUT_DIR = Path(r"D:/Experiments/2__scenes_all")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FFmpeg: H.264 NVENC output options / Параметры вывода FFmpeg для H.264 NVENC
FFMPEG_OUT_OPTS = [
    "-hide_banner",                        # EN: Cleaner console output / RU: Чище вывод в консоль 
    "-map", "0:v:0",                       # EN: Use only first video stream / RU: Берём только первый видеопоток
    "-vf", "scale=-2:720,format=yuv420p",  # EN: Scale to 720p height; enforce SDR 4:2:0 / RU: Масштаб до 720p; SDR 4:2:0
    "-c:v", "h264_nvenc",                  # EN: NVIDIA H.264 encoder / RU: Аппаратный кодек NVIDIA H.264
    "-preset", "p7",                       # EN: Slowest=best quality in NVENC / RU: Самое высокое качество (медленнее)
    "-tune", "lossless",                   # EN: Lossless mode *but* scaling+4:2:0 break true losslessness / RU: Режим без потерь, но масштаб+4:2:0 делают результат не полностью без потерь
    "-sn"                                  # EN: Drop subtitles / RU: Отключить субтитры
]

def get_video_fps(path: Path) -> float:
    """
    EN: Return video FPS for frame->time conversion. Raises if FPS is missing.
    RU: Возвращает FPS видео для пересчёта кадров в секунды. Бросает исключение, если FPS не получен.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps != fps:
        raise RuntimeError(f"Can't take FPS for: | Не удалось получить FPS для: {path}")
    return fps

def split_scene(input_path: Path, start_frame: int, end_frame: int, fps: float, out_path: Path):
    """
    EN:
    Cut [start_frame, end_frame) from input_path using FFmpeg, scale to 720p height and save to out_path
    with NVENC H.264. Uses -ss (input seek) for faster trimming and -t for duration.

    RU:
    Вырезает интервал [start_frame, end_frame) из input_path через FFmpeg, масштабирует до высоты 720
    и сохраняет в out_path кодеком NVENC H.264. Применяет -ss (входной seek) и -t для длительности.
    """
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss", f"{start_sec:.3f}",
        "-i", str(input_path),
        *FFMPEG_OUT_OPTS,
        "-t", f"{duration_sec:.3f}",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)

def process_video(video_path: Path):
    """
    EN:
    Detect scenes in video_path using TransNetV2, then export each scene as "video-{series}-{scene}.mp4"
    into OUTPUT_DIR. "series" is parsed from the 4th part of the filename split by underscores,
    otherwise falls back to the stem.

    RU:
    Определяет сцены в video_path через TransNetV2 и сохраняет каждую как "video-{series}-{scene}.mp4"
    в OUTPUT_DIR. "series" берётся из 4-й части имени файла (разделённого подчёркиваниями),
    иначе используется имя без расширения.
    """
    name = video_path.stem
    parts = name.split("_")
    series_num = parts[3] if len(parts) > 3 else name

    print(f"\n=== Processing | Обработка: {video_path.name} ===")
    scenes = predict_video(str(video_path), show_progressbar=True)
    print(f"Found scenes | Найдено сцен: {len(scenes)}")

    fps = get_video_fps(video_path)

    for idx, (start, end) in enumerate(scenes):
        out_fname = f"video-{series_num}-{idx:03d}.mp4"
        out_path = OUTPUT_DIR / out_fname
        print(f"  Scene | Сцена {idx:03d}: frames | кадры {start}–{end} → {out_fname}")
        split_scene(video_path, start, end, fps, out_path)

def main():
    """
    EN:
    Scan INPUT_DIR for MKV files, process each one, and report FFmpeg or unexpected errors
    without stopping the whole batch.

    RU:
    Ищет MKV в INPUT_DIR, обрабатывает каждый, и сообщает об ошибках FFmpeg или прочих,
    не прерывая обработку всех файлов.
    """
    files = sorted(glob.glob(str(INPUT_DIR / "*.mkv")))
    if not files:
        print(f"Нет MKV в папке: {INPUT_DIR}")
        return

    for f in files:
        try:
            process_video(Path(f))
        except subprocess.CalledProcessError as e:
            print(f"Error ffmpeg in processing | Ошибка ffmpeg при обработке {f}: {e}")
        except Exception as e:
            print(f"Unknown error | Непредвиденная ошибка для {f}: {e}")

if __name__ == "__main__":
    main()
