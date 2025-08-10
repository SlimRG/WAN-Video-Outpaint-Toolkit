# === Standard library / Стандартная библиотека ===
import os
import gc
from pathlib import Path

# === Typing / Типизация ===
from typing import Iterable, List, Optional, Dict, Tuple

# === Logging / Логирование ===
import logging
from logging.handlers import RotatingFileHandler
from tqdm.contrib.logging import logging_redirect_tqdm 

# === Core numeric & ML / Численные вычисления и ML ===
import numpy as np
import torch

# === Imaging & video IO / Обработка изображений и видео ===
import cv2
from PIL import Image
from tqdm import tqdm

# === Transformers: captioning & quantization / Трансформеры: подписи к кадрам и квантизация ===
from transformers import Blip2Processor, Blip2ForConditionalGeneration, UMT5EncoderModel

# === Accelerator / Ускоритель ===
from accelerate import Accelerator

# === HuggyFace Hub / Хранилище HuggyFace ===
from huggingface_hub import snapshot_download

# === Diffusers: autoencoders / Диффьюзеры: автоэнкодеры ===
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

# === Diffusers: transformer & pipeline / Диффьюзеры: трансформер и пайплайн ===
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel
from diffusers.pipelines.wan.pipeline_wan_vace import WanVACEPipeline

# === Diffusers: scheduler & utils / Диффьюзеры: планировщик и утилиты ===
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.export_utils import export_to_video
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

# ======
# === SETTINGS / НАСТРОЙКИ
# ======

# === Paths / Пути ===
INPUT_DIR  = "D:/Experiments/Video_Outpaint/2__scenes_all"
OUTPUT_DIR = "D:/Experiments/Video_Outpaint/3__outpaint"

# === Resolution & Canvas / Разрешение и холст ===
TARGET_WIDTH  = 1280
TARGET_HEIGHT = 720
SCALE_FACTOR = 1

# === Sampler & Guidance / Семплер и управление подсказкой ===
NUM_INFERENCE_STEPS = 9
GUIDANCE_SCALE      = 5.0
FLOW_SHIFT          = 5.0

# === Batching / Батчинг ===
BATCH_SIZE = 5

# === Prompts / Промпты ===
GRADIENT_WIDTH = 25
PROMPT = "anime, cyberpunk style"
NEGATIVE_PROMPT = (
    "blurry, lowres, noise, jpeg artifacts, color banding, chromatic aberration, "
    "haloing, oversharpened, edge stretching, edge warping, edge tearing, border seam, "
    "black border, white border, picture frame, padding, vignette, tiling, "
    "misaligned perspective, broken vanishing lines, bent guardrails, warped buildings, "
    "smeared bicycles, duplicated objects, extra limbs, mutated anatomy, temporal flicker, "
    "ghosting, popping, glossy highlights"
)

# === Device / Устройство ===
ACCELERATOR = Accelerator(
    mixed_precision="fp16"
)
DEVICE = ACCELERATOR.device

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure Python logging for console (and optional file).
    Настраивает логирование в консоль (и при желании в файл).
    """
    logger = logging.getLogger("outpaint")        # app logger / наш логгер приложения
    logger.setLevel(log_level)
    logger.propagate = False                      # no duplicate to root / не дублить в root

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # console handler / хендлер для консоли
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # optional file with rotation / опционально файл с ротацией
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=3)
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # HF libs verbosity / вербозность библиотек HF
    try:
        from diffusers.utils import logging as dlog
        dlog.set_verbosity_info()                 # info for diffusers / уровень info для diffusers
        dlog.enable_progress_bar()                
    except Exception:
        pass

    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_warning()              # warnings for transformers / предупреждения
        tlog.enable_default_handler()
        tlog.enable_explicit_format()
    except Exception:
        pass

    return logger


def download_models(cache_dir: str | None = None, logger: logging.Logger | None = None) -> Dict[str, str]:
    """
    Prefetch selected repos into the local HF cache (no model init); returns map repo->path /
    Предзагрузить выбранные репозитории в локальный кэш HF (без инициализации моделей); вернуть словарь repo->путь.
    """
    log = logger or logging.getLogger("outpaint")

    # Faster transport if available / Ускоренная загрузка при наличии
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        log.debug("Enabled HF transfer acceleration / Включена акселерация HF transfer")

    repos: List[Tuple[str, List[str]]] = [
        ("black-forest-labs/FLUX.1-Fill-dev", ["vae/*"]),
        ("Salesforce/blip2-flan-t5-xl", ["*"]),
        ("Wan-AI/Wan2.1-VACE-14B-diffusers", ["*"]),
        ("QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF", ["Wan2.1_T2V_14B_FusionX_VACE-Q6_K.gguf"]),
        ("city96/umt5-xxl-encoder-gguf", ["umt5-xxl-encoder-Q8_0.gguf"]),
    ]

    log.info("Downloading %d repo(s) to cache_dir=%s / Загрузка %d репозитори(я/ев) в %s",
             len(repos), cache_dir or "<HF default>", len(repos), cache_dir or "<HF по умолчанию>")

    out_paths: Dict[str, str] = {}

    for repo_id, patterns in repos:
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=patterns,   # only needed parts / только нужные части
                local_dir=cache_dir,       # optional target dir / опциональная директория
                local_dir_use_symlinks=True
            )
            out_paths[repo_id] = local_path
        except Exception:
            log.exception("Failed to download %s / Ошибка загрузки %s", repo_id, repo_id)
            raise  # re-raise so caller can handle / пробрасываем дальше для обработки наверху

    log.info("All snapshots are ready / Все снапшоты готовы")
    return out_paths

def find_video_files(
    directory: str | Path,
    extensions: Optional[Iterable[str]] = None,
    recursive: bool = True,
    absolute: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """
    Find all video files under a directory and return their paths /
    Найти все видеофайлы в каталоге и вернуть их пути.

    Args / Параметры:
        directory: root directory to search in /
                   корневая директория для поиска
        extensions: iterable of file extensions (with or without dot), case-insensitive;
                    if None, uses a common default set /
                    набор расширений (с/без точки), регистронезависимо; если None — используется типовой список
        recursive: search subdirectories if True /
                   искать во вложенных каталогах, если True
        absolute: return absolute paths if True, else relative to `directory` /
                  возвращать абсолютные пути, иначе относительные к `directory`
        logger: optional logger for diagnostics /
                необязательный логгер для диагностики

    Returns / Возврат:
        list of Path objects for matched video files /
        список объектов Path для найденных видеофайлов
    """
    log = logger or logging.getLogger("outpaint")

    root = Path(directory)
    if not root.exists():
        log.warning("Directory not found: %s / Каталог не найден: %s", root, root)
        return []

    # Default set of common video extensions / Набор типовых видео-расширений
    if extensions is None:
        extensions = {
            "mp4", "m4v", "mov", "mkv", "avi", "wmv", "flv", "webm",
            "mpg", "mpeg", "mts", "m2ts", "ts", "vob", "3gp", "3g2",
            "ogv", "mxf"
        }

    # Normalize extensions to dotted lowercase (e.g., ".mp4") /
    # Нормализуем расширения к виду с точкой и в нижнем регистре (например, ".mp4")
    exts = {"." + ext.lower().lstrip(".") for ext in extensions}

    # Choose iterator based on recursion flag /
    # Выбираем итератор в зависимости от флага рекурсии
    it = root.rglob("*") if recursive else root.iterdir()

    matches: List[Path] = []
    for p in it:
        # Skip non-files / Пропускаем не-файлы
        if not p.is_file():
            continue
        # Check extension / Проверяем расширение
        if p.suffix.lower() in exts:
            matches.append(p if absolute else p.relative_to(root))

    # Sort for stable order / Сортируем для стабильного порядка
    matches.sort()

    log.info(
        "Found %d video file(s) in %s / Найдено %d видеофайлов в %s",
        len(matches), root, len(matches), root
    )
    return matches

def read_video_frames(input_path, logger: logging.Logger | None = None):
    """
    Read all frames from a video file with a progress bar; returns frames and FPS /
    Считать все кадры из виде файла с прогресс-баром; вернуть кадры и FPS.

    Args / Параметры:
        input_path (str): path to the input video file /
                          путь к входному видеофайлу
        logger (logging.Logger | None): optional logger for diagnostics /
                                        необязательный логгер для диагностики

    Returns / Возврат:
        (list[PIL.Image.Image], int): list of RGB frames and integer FPS /
                                      список кадров RGB и целочисленный FPS
    """
    log = logger or logging.getLogger("outpaint")  # use provided or fallback logger / используем переданный или резервный логгер
    frames = []
    cap = None

    try:
        cap = cv2.VideoCapture(input_path)  # open video / открыть видео
        if not cap.isOpened():
            log.error("Failed to open video: %s / Не удалось открыть видео: %s", input_path, input_path)
            return frames, 30  # safe default / безопасный дефолт

        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # read FPS or fallback / читаем FPS или дефолт
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # total frames if known / общее число кадров, если известно
        log.info("Reading video: %s (fps=%.2f, total=%s) / Чтение видео: %s (fps=%.2f, всего=%s)",
                 input_path, fps, total if total > 0 else "unknown", input_path, fps, total if total > 0 else "неизвестно")

        # progress bar with known total if available / прогресс-бар с известным total, если доступен
        with tqdm(total=total if total > 0 else None, desc="Reading video frames") as pbar:
            while True:
                ret, frame_bgr = cap.read()  # read next frame / читаем следующий кадр
                if not ret:
                    break  # EOF or error / конец файла или ошибка

                # BGR -> RGB and to PIL / BGR -> RGB и в PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                pbar.update(1)

        log.info("Frames read: %d / Считано кадров: %d", len(frames), len(frames))
        return frames, int(fps)

    except Exception:
        log.exception("Error while reading video / Ошибка при чтении видео")
        raise  # bubble up / пробрасываем выше
    finally:
        if cap is not None:
            cap.release()  # ensure release / гарантированно освобождаем ресурс

def make_base_canvases(
    target_w: int,
    target_h: int,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> List[Image.Image]:
    """
    Generate `batch_size` noise canvases using FLUX VAE once at startup /
    Сгенерировать `batch_size` шумовых холстов через FLUX VAE один раз при старте.
    
    Returns / Возврат:
        list[PIL.Image.Image]: list of noise canvases at (target_w, target_h) /
                               список шумовых холстов размера (target_w, target_h)
    """
    log = logger or logging.getLogger("outpaint")
    base_canvases: List[Image.Image] = []

    single_vae = None
    try:
        # Load VAE directly on CUDA to warm kernels / Грузим VAE сразу на CUDA для прогрева ядер
        single_vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map={"": "cuda:0"},
        )

        vae_scale = 8  # FLUX latent downscale factor / коэффициент даунскейла латентов FLUX
        latent_channels = single_vae.config["latent_channels"]
        scaling = single_vae.config["scaling_factor"]

        # Generate N canvases / Генерируем N холстов
        h_lat = target_h // vae_scale
        w_lat = target_w // vae_scale

        with torch.no_grad():
            for _ in tqdm(range(batch_size), desc="Generating VAE noise"):
                # Sample latent ~ N(0,1) * scaling / Семплируем латент ~ N(0,1) * scaling
                base_noise = torch.randn(
                    1,
                    latent_channels,
                    h_lat,
                    w_lat,
                    device=device,
                    dtype=single_vae.dtype,
                ) * scaling

                # Decode to image space / Декодируем в пространство изображений
                decoded = single_vae.decode(base_noise).sample[0]  # type: ignore
                noise_img = (
                    (decoded / 2 + 0.5).clamp(0, 1)  # normalize [-1,1]→[0,1] / нормализация
                    .permute(1, 2, 0)                # (H,W,C)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

                # Convert to PIL / Преобразуем в PIL
                base_canvases.append(Image.fromarray((noise_img * 255).astype(np.uint8)))

        log.info("Generated %d noise canvases / Сгенерировано %d шумовых холстов", len(base_canvases), len(base_canvases))
        return base_canvases
    except Exception:
        log.exception("Failed to generate base canvases / Ошибка генерации шумовых холстов")
        raise
    finally:
        # Cleanup VRAM / Очистка видеопамяти
        if single_vae is not None:
            del single_vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def make_canvases_and_masks(
    frames: List[Image.Image],
    base_canvases: List[Image.Image],
    target_w: int,
    target_h: int,
    batch_size: int,
    grad_width: int = 25,
    logger: logging.Logger | None = None,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Compose frames onto precomputed noise canvases and build a single horizontal feather mask /
    Наложить кадры на заранее сгенерированные шумовые холсты и создать одну горизонтальную маску с растушёвкой.

    Mask convention / Принцип маски:
      0 — keep/original region (no edit) /
      0 — сохраняем исходник (без редактирования)
      255 — outpaint region (to be filled) /
      255 — область дорисовки (outpaint)

    Args / Параметры:
      frames: list of input frames (PIL) / список входных кадров (PIL)
      base_canvases: list of noise canvases to paste onto / список шумовых холстов для наложения
      target_w, target_h: output size / размер выходного холста
      batch_size: size of generation batch (used to cycle canvases) / размер батча (для циклического выбора холстов)
      grad_width: feather width (px) at left/right edges / ширина растушёвки (пикс) слева/справа
      logger: optional logger / необязательный логгер

    Returns / Возвращает:
      (canvases, masks): lists aligned with `frames` / списки, выровненные по `frames`
    """
    log = logger or logging.getLogger("outpaint")

    canvases: List[Image.Image] = []
    masks: List[Image.Image] = []
    bc_index = 0

    for img in tqdm(frames, total=len(frames), desc="Compositing canvases"):
        w, h = img.size
        dx = (target_w - w) // 2
        dy = (target_h - h) // 2

        # Pick and copy a base noise canvas / Берём и копируем шумовой холст
        canvas = base_canvases[bc_index].copy()
        canvas.paste(img, (dx, dy))  # paste original frame in the center / кладём исходный кадр по центру
        canvases.append(canvas)

        # Cycle base canvas index within batch_size / Циклический индекс холстов в пределах batch_size
        bc_index += 1
        if bc_index == batch_size:
            bc_index = 0

        # Build the mask once, then reuse via copy / Маску создаём один раз и далее копируем
        if len(masks) == 0:
            # Start with full-255 (outpaint everywhere) / Начинаем с 255 (дорисовывать везде)
            mask_arr = np.full((target_h, target_w), 255, dtype=np.uint8)

            # Put original frame area to 0 (keep) / Область исходного кадра → 0 (оставляем)
            mask_arr[dy:dy + h, dx:dx + w] = 0

            # Horizontal feather towards the center at both sides /
            # Горизонтальная растушёвка к центру по обоим бокам
            # 255 → 0 from border to inside / 255 → 0 от границы к центру
            grad = (np.linspace(1, 0, grad_width) * 255).astype(np.uint8)

            # Left edge / Левый край
            mask_arr[dy:dy + h, dx:dx + grad_width] = grad[None, :]

            # Right edge (mirrored) / Правый край (зеркально)
            mask_arr[dy:dy + h, dx + w - grad_width:dx + w] = grad[::-1][None, :]

            mask = Image.fromarray(mask_arr, mode="L")
            masks.append(mask)
        else:
            # Reuse mask for subsequent frames / Переиспользуем маску для остальных кадров
            masks.append(masks[0].copy())

    log.info(
        "Canvases & masks ready: %d items / Готово холстов и масок: %d шт.",
        len(canvases), len(canvases)
    )
    return canvases, masks

def generate_prompts_with_blip(
    frames: List[Image.Image],
    batch_size: int,
    base_prompt: str,
    device: str,
    logger: logging.Logger | None = None,
) -> List[str]:
    """
    Generate per-chunk captions with BLIP2 and prepend a base prompt /
    Сгенерировать подписи BLIP2 для кадров (по шагу) и добавить базовый промпт.

    Strategy / Стратегия:
      - take every (batch_size-1)-th frame to reduce calls /
        берём каждый (batch_size-1)-й кадр, чтобы сократить число вызовов
      - run a single batched BLIP2 generate /
        запускаем единый батчевый вызов BLIP2
      - return list of prompts aligned to selected frames (not all frames) /
        возвращаем список подсказок, соответствующих выбранным кадрам (не всем кадрам)
    """
    log = logger or logging.getLogger("outpaint")
    prompts: List[str] = []

    if not frames:
        log.warning("No frames provided to BLIP2 / В BLIP2 не переданы кадры")
        return prompts

    # Pick frames for captioning / Выбираем кадры для описания
    step = max(1, batch_size - 1)
    img4prompt = frames[0:len(frames):step]

    # Load BLIP2 / Загружаем BLIP2
    try:
        proc_blip = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            use_fast=True,
            torch_dtype=torch.bfloat16,   # tokenizer/processor dtype hint / подсказка dtype
        )
        model_blip = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.bfloat16,
            device_map=None               # no accelerate hooks / без accelerate-хуков
        ).to(device).eval()

        # Encode & generate / Кодируем и генерируем
        with torch.no_grad():
            inputs = proc_blip(images=img4prompt, return_tensors="pt").to(device)
            # default generate is fine; customize if needed (num_beams, max_new_tokens) /
            # стандартного generate достаточно; при желании настраивайте (num_beams, max_new_tokens)
            output_ids = model_blip.generate(**inputs) # type: ignore
            texts = proc_blip.batch_decode(output_ids, skip_special_tokens=True)

        base_prompt = (base_prompt or "").strip()
        for text in texts:
            caption = (text or "").strip()
            # Compose final prompt / Собираем итоговую подсказку
            if base_prompt and caption:
                prompts.append(f" {base_prompt}, {caption}")
            elif base_prompt:
                prompts.append(f" {base_prompt}")
            else:
                prompts.append(f" {caption}")

        log.info(
            "BLIP2: generated %d prompt(s) / BLIP2: сгенерировано %d промпт(ов)",
            len(prompts), len(prompts)
        )
        return prompts

    except Exception:
        log.exception("BLIP2 captioning failed / Сбой при генерации подписей BLIP2")
        raise
    finally:
        # Cleanup VRAM/CPU / Очистка VRAM/CPU
        try:
            del model_blip
        except Exception:
            pass
        try:
            del proc_blip
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_wan_pipeline(
    flow_shift: float,
    logger: Optional[logging.Logger] = None,
) -> WanVACEPipeline:
    """
    Build WAN VACE pipeline with 4-bit BnB transformer and SAGE attention; returns ready pipe /
    Собрать пайплайн WAN VACE с 4-битным трансформером BnB и вниманием SAGE; вернуть готовый pipe.
    
    Notes / Примечания:
      - No `device_map` is used; you may move modules to CUDA outside if needed /
        `device_map` не используется; при необходимости перенесите модули на CUDA снаружи.
      - Do NOT mix BnB with `enable_model_cpu_offload()` (performance/compat issues) /
        Не смешивайте BnB с `enable_model_cpu_offload()` (просадка и возможные конфликты).
      - Use `"sage"` (not `"sage_varlen"`) for stable outpaint & masks /
        Используйте `"sage"` (не `"sage_varlen"`) для стабильной работы масок в outpaint.
    """
    log = logger or logging.getLogger("outpaint")

    # --- VAE / VAE ---
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-VACE-14B-diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # --- Text Encoder / Кодировщик текста ---
    # [TODO - 0001] No support at 10.08.2025 - https://github.com/huggingface/transformers/issues/40067
    # text_encoder = UMT5EncoderModel.from_pretrained(
    #     "city96/umt5-xxl-encoder-gguf",
    #     gguf_file="umt5-xxl-encoder-Q8_0.gguf",
    #     torch_dtype=torch.float16,
    # )

    # --- Transformer / Трансформер ---
    transformer = WanVACETransformer3DModel.from_single_file(
        "https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF/blob/main/Wan2.1_T2V_14B_FusionX_VACE-Q6_K.gguf",
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.float16
        ),
        torch_dtype=torch.float16,
    )

    # --- Pipeline assembly / Сборка пайплайна ---
    pipe = WanVACEPipeline.from_pretrained(
        "Wan-AI/Wan2.1-VACE-14B-diffusers",
        vae=vae,
        #text_encoder=text_encoder, # [TODO - 0001]
        transformer=transformer,
        torch_dtype=torch.float16
    )

    # --- Scheduler / Планировщик ---
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

    # Sage
    pipe.transformer.set_attention_backend("sage")  # stable for masks / стабильно для масок
    log.info("Attention backend set to 'sage' / Бэкенд внимания установлен: 'sage'")

    # memory-friendly encode/decode / экономит память при encode/decode
    pipe.vae.enable_tiling()  

    # VAE memory save
    pipe.enable_model_cpu_offload()

    log.info("WAN pipeline is ready / Пайплайн WAN готов")
    return pipe

def run_outpaint(
    pipe,
    canvases: List[Image.Image],
    masks: List[Image.Image],
    prompts: List[str],
    batch_size: int,
    target_w: int,
    target_h: int,
    steps: int,
    guidance: float,
    negative_prompt: str,
    fps: int,
    scale_factor,
    output_path: str,
    logger: logging.Logger | None = None,
) -> None:
    """
    Run WAN outpaint over frames in overlapping batches and write the result video /
    Выполнить outpaint WAN по кадрам пакетами с перекрытием и сохранить результат в видео.

    Mask convention / Принцип маски:
      0   — keep original pixels (no edit) /
            оставить исходные пиксели (без редактирования)
      255 — fill/outpaint region /
            дорисовывать в этой области
    """
    log = logger or logging.getLogger("outpaint")

    if not canvases or not masks:
        log.error("Empty canvases or masks / Пустые списки холстов или масок")
        return

    if batch_size < 2:
        log.error("batch_size must be >= 2 / batch_size должен быть >= 2")
        return

    # Sanity for prompts: repeat last if shorter / Проверка промптов: повторяем последний, если не хватает
    def get_prompt(i: int) -> str:
        return prompts[i] if i < len(prompts) else prompts[-1]

    result_frames: List[np.ndarray] = []
    idx = 0
    prev_frame = canvases[0]
    prev_mask = masks[0]
    black_mask = Image.new("L", (target_w, target_h), 0)  # do not edit previous frame / не редактировать прошлый кадр

    try:
        for i in tqdm(range(0, len(canvases), batch_size - 1), desc="Outpainting batches"):
            # Build batch with overlap / Формируем батч с перекрытием
            batch_canv = [prev_frame] + canvases[i : i + batch_size - 1]
            batch_mask = [prev_mask]  + masks[i    : i + batch_size - 1]
            batch_len = len(batch_canv)

            # Do not re-edit the first (prev) frame except for the very first batch /
            # Не редактируем первый (prev) кадр, кроме самого первого батча
            if i != 0:
                batch_mask[0] = black_mask

            # Pad to full batch / Добиваем до полного батча
            if batch_len < batch_size:
                last_c, last_m = batch_canv[-1], batch_mask[-1]
                pad = batch_size - batch_len
                batch_canv.extend([last_c] * pad)
                batch_mask.extend([last_m] * pad)

            prompt = get_prompt(idx)

            # Inference / Инференс
            with torch.no_grad():
                out = pipe(
                    video=batch_canv,
                    mask=batch_mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=int(target_h*scale_factor),
                    width=int(target_w*scale_factor),
                    num_frames=batch_size,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                ).frames[0]

            # Drop the overlapped first frame / Отбрасываем первый «контекстный» кадр
            out = out[1:batch_len]
            result_frames.extend(out)
            idx += 1

            # Prepare prev_* for next batch / Готовим prev_* для следующего батча
            prev_frame_np = out[batch_len - 2]
            if prev_frame_np.ndim == 4 and prev_frame_np.shape[0] == 1:
                prev_frame_np = prev_frame_np.squeeze(0)
            if prev_frame_np.ndim == 3 and prev_frame_np.shape[0] in (1, 3):
                prev_frame_np = np.transpose(prev_frame_np, (1, 2, 0))
            if prev_frame_np.dtype != np.uint8:
                prev_frame_np = (prev_frame_np * 255).clip(0, 255).astype(np.uint8)
            prev_frame = Image.fromarray(prev_frame_np)

            scale = 1 / scale_factor
            w, h = prev_frame.size
            new_size = (round(w * scale), round(h * scale))
            prev_frame = prev_frame.resize(new_size, Image.Resampling.LANCZOS)

            prev_mask  = batch_mask[batch_len - 2]

        # Write video / Сохраняем видео
        export_to_video(result_frames, output_path, fps=int(fps), quality=10)
        log.info(
            "Outpaint done → %s (frames=%d) / Готово → %s (кадров=%d)",
            output_path, len(result_frames), output_path, len(result_frames)
        )

    except Exception:
        log.exception("Outpaint failed / Сбой в процессе outpaint")
        raise

def main():
    # Set logging / Устанавливаем логирование
    logger = setup_logging()

    # Download models / Загрузка моделей
    download_models(logger=logger)

    # Find video / Поиск видео
    video_paths = find_video_files(directory=INPUT_DIR, recursive=False, logger=logger)

    for video_path in video_paths:
        # Get frames / Раскадровка
        frames, fps = read_video_frames(video_path, logger=logger)

        # Generate base noise canvases / Генерация шумовых основ
        base_canvases = make_base_canvases(TARGET_WIDTH, TARGET_HEIGHT, BATCH_SIZE, DEVICE, logger)

        # Generate noisy imgs and masks / Генерация зашумленных изображений и масок
        canvases, masks = make_canvases_and_masks(
            frames, base_canvases, TARGET_WIDTH, TARGET_HEIGHT, BATCH_SIZE, GRADIENT_WIDTH, logger
        )

        # Generate BLIP2 prompts / Генерация BLIP2 описаний
        prompts = generate_prompts_with_blip(frames, BATCH_SIZE, PROMPT, DEVICE, logger)

        # Generate WAN pipeline / Генерация WAN пайплайна
        pipe = build_wan_pipeline(FLOW_SHIFT, logger)

        # Get output path / Получения пути извлечения
        output_path = os.path.join(OUTPUT_DIR, f"{Path(video_path).stem}.mkv")

        # Outpaint / Дорисовка кадров
        run_outpaint(
            pipe=pipe,
            canvases=canvases,
            masks=masks,
            prompts=prompts,
            batch_size=BATCH_SIZE,
            target_w=TARGET_WIDTH,
            target_h=TARGET_HEIGHT,
            steps=NUM_INFERENCE_STEPS,
            guidance=GUIDANCE_SCALE,
            negative_prompt=f" {NEGATIVE_PROMPT}",
            fps=fps,
            output_path=output_path,
            scale_factor=SCALE_FACTOR,
            logger=logger
        )

    logger.info(f"Done! / Готово!")

if __name__ == "__main__":
    main()
