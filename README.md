# WAN 2.1 Video Outpaint Toolkit

**EN / RU README**

> **EN:** Two-stage pipeline: (1) split videos into scenes with TransNetV2 + FFmpeg (NVENC), (2) outpaint scenes with WAN 2.1 VACE (FusionX) via Diffusers.
> **RU:** Двухэтапный конвейер: (1) разбиение видео на сцены TransNetV2 + FFmpeg (NVENC), (2) дорисовка сцен WAN 2.1 VACE (FusionX) через Diffusers.

---

## Contents / Состав

* `1__sceneSeparator.py` — **EN:** scene detection + export per-scene MP4 (NVENC, 720p). **RU:** детект сцен + экспорт отдельных MP4 (NVENC, 720p).
* `2__Outpaint.py` — **EN:** reads scenes, builds masks, BLIP2 captions, runs WAN outpaint, writes video. **RU:** читает сцены, строит маски, генерирует подписи BLIP2, запускает WAN outpaint, пишет видео.

```
repo/
├─ 1__sceneSeparator.py
├─ 2__Outpaint.py
└─ README.md  (this file / этот файл)
```

---

## Requirements / Требования

**Hardware / Железо**

* **NVIDIA GPU** with recent driver; **NVENC** for H.264 export.
* 12–24 GB VRAM recommended for comfortable outpaint (less works with offload/quant).
* Fast NVMe helps model caching.

**Software / Софт**

* Python **3.10+**
* FFmpeg build **with NVENC** (`h264_nvenc`)
* CUDA-enabled PyTorch

**Python packages / Пакеты Python**

```
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow numpy tqdm accelerate transformers diffusers huggingface_hub
pip install transnetv2pt
```

> **Windows + FFmpeg:** verify `ffmpeg -hide_banner -encoders | findstr nvenc` shows `h264_nvenc`.
> **Windows + FFmpeg:** проверьте, что `ffmpeg -hide_banner -encoders | findstr nvenc` показывает `h264_nvenc`.

---

## Quick Start / Быстрый старт

1. **Put source videos** into `D:/Experiments/1__original` (`*.mkv`).
   **Сложите исходные видео** в `D:/Experiments/1__original` (`*.mkv`).

2. **Split scenes**:

```bash
python 1__sceneSeparator.py
```

Outputs to `D:/Experiments/2__scenes_all/` as `video-{series}-{scene}.mp4`.

3. **Outpaint**:

```bash
python 2__Outpaint.py
```

Reads scenes from `2__scenes_all`, writes outpainted videos to `D:/Experiments/Video_Outpaint/3__outpaint`.

---

## How it works / Как это работает

### 1) Scene split (`1__sceneSeparator.py`)

* **EN:** TransNetV2 finds \[start\_frame, end\_frame) per scene. Each scene is cut with FFmpeg using input `-ss` and `-t`. Output is **H.264 NVENC**, **scaled to 720p height**, format **yuv420p**.
* **RU:** TransNetV2 находит интервалы сцен. Каждая сцена режется FFmpeg через входной `-ss` и `-t`. Выход — **H.264 NVENC**, **масштаб до 720p по высоте**, формат **yuv420p**.

> **Note / Важно:** with `scale` + `yuv420p` video is **not truly lossless** despite `-tune lossless`.
> При `scale` + `yuv420p` видео **не полностью без потерь**, несмотря на `-tune lossless`.

### 2) Outpaint (`2__Outpaint.py`)

* **EN:** Loads/warms models, creates **noise canvases** (FLUX VAE), pastes original frames centered on a bigger canvas (TARGET\_W×TARGET\_H). Builds a **single horizontal feather mask**: `0=keep`, `255=fill`. Optional **BLIP2** captions merge with your base `PROMPT`. WAN VACE runs in **overlapping batches** (`BATCH_SIZE`) to keep temporal context.
* **RU:** Загружает модели, создает **шумовые холсты** (FLUX VAE), вставляет исходные кадры по центру увеличенного холста (TARGET\_W×TARGET\_H). Строит **горизонтальную маску с растушёвкой**: `0=оставить`, `255=дорисовать`. Опциональные подписи **BLIP2** объединяются с базовым `PROMPT`. WAN VACE работает **батчами с перекрытием** (`BATCH_SIZE`) для сохранения контекста.

---

## Key Settings / Ключевые настройки

**In `1__sceneSeparator.py`**

* `FFMPEG_OUT_OPTS`:

  * `-vf "scale=-2:720,format=yuv420p"` — **EN:** keep width multiple of 2, 720p height. **RU:** ширина кратна 2, высота 720p.
  * `-c:v h264_nvenc -preset p7 -tune lossless` — **EN:** slow/high quality; not truly lossless due to scale/4:2:0. **RU:** медленно/качественно; не полностью без потерь из-за масштаба/4:2:0.

**In `2__Outpaint.py`**

* **Resolution / Разрешение:** `TARGET_WIDTH`, `TARGET_HEIGHT`, `SCALE_FACTOR` (internal up/down, resampled back).
* **Diffusion params / Параметры диффузии:**

  * `NUM_INFERENCE_STEPS` (e.g., 9..32) — speed/quality trade-off.
  * `GUIDANCE_SCALE` (e.g., 4–6) — prompt adherence vs flicker.
  * `FLOW_SHIFT` (e.g., 4–6) — sampler’s denoise flow tweak.
* **Batching:** `BATCH_SIZE` (e.g., 5). Larger batch = more VRAM, fewer calls.
* **Mask feather / Растушёвка маски:** `GRADIENT_WIDTH` (e.g., 25 px).
* **Prompts:** `PROMPT`, `NEGATIVE_PROMPT`. BLIP2 adds captions per chunk.

---

## Models & Downloads / Модели и загрузка

`2__Outpaint.py` calls `download_models()` to prefetch:

* FLUX.1-Fill VAE (for noise canvases)
* BLIP2 (captioning)
* WAN 2.1 VACE diffusers + Quantized **FusionX** GGUF

**Cache / Кэш:** uses Hugging Face cache. Enable fast transport automatically.
**Кэш:** использует кэш HF. Быстрая загрузка включается автоматически.

---

## Speed vs VRAM / Скорость vs память

* **Fast profile idea / Быстрый профиль (идея):**

  * Keep models on GPU (no `enable_model_cpu_offload()`), prefer **fp16** VAE, smaller **steps** (e.g., 28–32 for higher quality, ≤16 for speed), moderate **guidance** (4.5–5.5).
* **Safe profile / Безопасный профиль:**

  * Enable CPU offload (slower, but fits).
* **Quantization / Квантизация:**

  * Use **Q6\_K** (default in code) or reduce to **Q5\_K/Q4\_K** if VRAM tight.
* **Attention backend / Бэкенд внимания:**

  * `sage` is stable for masks; if running **without** masks, `sdpa` can be faster (toggle before calling).
* **Windows env tweak / Настройка окружения:**

  * `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce VRAM fragmentation.

> The provided script currently uses `enable_model_cpu_offload()` for VAE to save VRAM. You can comment it out for speed if memory allows.
> Скрипт сейчас включает `enable_model_cpu_offload()` для экономии VRAM. Для скорости можно отключить, если памяти хватает.

---

## Usage Notes / Замечания по использованию

* **Series id parsing / Парсинг series:** `1__sceneSeparator.py` takes the **4th** underscore-separated token; otherwise uses filename stem.
* **VFR caveat / Переменный FPS:** cutting by frames assumes stable FPS; VFR sources may need time-accurate options (`-accurate_seek`) or pre-conversion.
* **Not truly lossless / Не полностью без потерь:** scaling + 4:2:0 implies loss even with `-tune lossless`.
* **Masks convention / Конвенция масок:** `0 = keep`, `255 = outpaint`.

---

## Troubleshooting / Диагностика

* **`h264_nvenc` not found:** install FFmpeg with NVENC; update NVIDIA driver.
  **`h264_nvenc` не найден:** поставьте FFmpeg с NVENC; обновите драйвер NVIDIA.
* **CUDA OOM:** lower `BATCH_SIZE`, reduce `TARGET_*`, switch GGUF to `Q5_K/Q4_K`, enable CPU offload.
  **Недостаточно VRAM:** уменьшите `BATCH_SIZE`, `TARGET_*`, снижайте квант, включайте оффлоад.
* **Slow first run:** model download/JIT warmup; later runs are faster.
  **Медленный первый запуск:** скачивание моделей/прогрев; дальше быстрее.
* **Flicker / мерцание:** reduce `GUIDANCE_SCALE`, keep `FLOW_SHIFT` moderate, ensure consistent prompts.
  **Мерцание:** снизьте `GUIDANCE_SCALE`, держите `FLOW_SHIFT` умеренным, стабилизируйте промпты.

---

## Customization / Кастомизация

* **Paths:** change `INPUT_DIR`, `OUTPUT_DIR` in both scripts as needed.
* **Captioning:** disable BLIP2 by passing a static `PROMPT` and skipping `generate_prompts_with_blip`.
* **Attention backend:** `pipe.transformer.set_attention_backend("sage" | "sdpa")` per use-case.
* **Export codec:** adjust `FFMPEG_OUT_OPTS` (e.g., HEVC `hevc_nvenc`, different scale, CRF-based SDR, etc.).

---

## License / Лицензия

* **Code:** your repository’s license (add one).
* **Models:** follow respective model licenses on Hugging Face.
* **Код:** добавьте лицензию репозитория.
* **Модели:** соблюдайте лицензии моделей на Hugging Face.

---

## Acknowledgements / Благодарности

* TransNetV2 (scene boundary detection)
* Hugging Face Diffusers / Transformers
* WAN 2.1 VACE (FusionX) community work
* FFmpeg + NVENC

---

### Short Checklist / Короткий чек-лист

* [ ] Python 3.10+, CUDA PyTorch, FFmpeg with NVENC
* [ ] Put `*.mkv` to `D:/Experiments/1__original`
* [ ] `python 1__sceneSeparator.py` → scenes in `2__scenes_all`
* [ ] `python 2__Outpaint.py` → outpainted in `3__outpaint`
* [ ] Tune `NUM_INFERENCE_STEPS`, `GUIDANCE_SCALE`, `FLOW_SHIFT`, `BATCH_SIZE` as needed

Если нужно — сделаю отдельные **EN-only** и **RU-only** варианты README или добавлю раздел «Advanced tuning» с готовыми профилями под вашу видеокарту.
