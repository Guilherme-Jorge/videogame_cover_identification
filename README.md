# Videogame Cover Identification

A simple system that identifies a videogame by its cover. Give it a photo of a game case or a cropped cover, and it returns the most likely match from a local database, along with confidence signals and alternatives.

It works by:
- Embedding every cover image into a vector space using a CLIP-based encoder
- Building a FAISS index for fast nearest-neighbor search
- Detecting and rectifying the cover from your photo (perspective fix)
- Retrieving top candidates and re-ranking them using a geometric SIFT score

## Key features
- **Cover detection and rectification**: Finds the largest plausible quadrilateral and perspective-corrects it.
- **Fast retrieval**: FAISS inner-product index over L2-normalized embeddings.
- **CLIP-based encoder**: Uses OpenCLIP, with optional fine-tuning for this task.
- **Confidence signals**: Cosine similarity and geometric matching score.
- **Simple CLI**: Build the index and run a search with one command each.

## What’s included
- A ready-to-use dataset folder structure under `data/` (placeholders may already exist):
  - `data/metadata.jsonl`: One JSON object per line with fields like `id`, `name`, `cover_id`, `cover_url`, `local_filename`.
  - `data/covers/`: Image files referenced by `local_filename`.
  - `data/covers.faiss`, `data/covers.npy`, `data/covers_meta.json`: Built artifacts after indexing.
  - `data/cover_encoder.pt`: Optional fine-tuned weights file.
- Example input images in `samples/`.

## Requirements
- Python 3.12+
- PyTorch and torchvision (CUDA 12.1 wheels supported in `requirements.txt`)
- OpenCV, FAISS (CPU), OpenCLIP, NumPy, Pillow, TQDM

Install dependencies with either file:
```bash
# Option A: use requirements.txt (CPU or non-CUDA builds)
pip install -r requirements.txt

# Option A (CUDA 12.1, +cu121 wheels): add the extra index and strategy
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
            --index-strategy unsafe-best-match \
            -r requirements.txt

# Option B: use pyproject.toml (requires pip>=23 or uv/pdm/poetry)
pip install .

# Option B (CUDA 12.1, +cu121 wheels): add the extra index and strategy
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
            --index-strategy unsafe-best-match \
            .
```
Note: If using CUDA, ensure your PyTorch install matches your system. The `requirements.txt` pins CUDA 12.1 builds; adjust if needed.

## Data format
The system expects a JSONL metadata file describing each cover:
```json
{"id": 7346, "name": "Some Game", "cover_id": "abc123", "cover_url": "https://...", "local_filename": "covers/7346.jpg"}
```
- `local_filename` can be relative (resolved under a detected covers root) or absolute.
- Images should live under a `covers/` directory.

By default, paths are detected automatically. You can also set an environment variable to control it:
```bash
export COVERS_ROOT=/absolute/path/to/data
```
The code looks for a `covers/` folder under this root.

## Data source
This project is based on cover data from the IGDB database ([IGDB](https://www.igdb.com/)).

## Build the index
Turn your covers into embeddings and create the FAISS index:
```bash
python -m src.build_index
```
Default inputs and outputs come from `src/config.py`:
- Reads `metadata.jsonl`
- Writes `covers.faiss`, `covers.npy`, `covers_meta.json`
- Uses GPU for indexing if available (configurable)

You can change file paths and options by editing `src/config.py` or by calling `build_index` from Python with custom arguments.

## Run a search
Provide a path to a photo or a cropped cover:
```bash
python -m src.search_cover /path/to/photo.jpg
```
What it does:
- Detects and rectifies the cover (and exports `export_crop.png` and `export_debug.png`).
- Embeds both the rectified crop and the full image.
- Retrieves candidates from FAISS and re-ranks the top results with SIFT.
- Chooses the stronger of the two strategies (crop vs full) based on combined signals.

The CLI prints a JSON result like:
```json
{
  "confident": true,
  "strategy": "crop",
  "cosine": 0.41,
  "geom_score": 0.62,
  "match": {
    "id": 7346,
    "name": "Some Game",
    "cover_id": "abc123",
    "cover_url": "https://...",
    "local_filename": "covers/7346.jpg"
  },
  "alternatives": [
    {"cosine": 0.39, "id": 101, "name": "Another Game"}
  ],
  "other_variant": {
    "strategy": "full",
    "cosine": 0.37,
    "geom_score": 0.21
  },
  "exports": {
    "crop": "export_crop.png",
    "debug": "export_debug.png",
    "quad": [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
  }
}
```

## Configuration
Centralized in `src/config.py`:
- **Model**: CLIP backbone (`ViT-B-16` by default), optional fine-tuned weights path `data/cover_encoder.pt`.
- **Search**: `topk`, `rerank_k`, `accept_threshold`, `geom_score_threshold`.
- **Index**: default paths and GPU usage.
- **Training**: epochs, batch size, LR, output path, AMP mode.
- **Paths**: `COVERS_ROOT` env var and default directories.
- **Logging**: level and format.

## Training (optional)
You can fine-tune the image encoder on your covers to improve retrieval robustness.

Command-line entry:
```bash
python -m src.train_finetune \
  --jsonl data/metadata.jsonl \
  --root /path/to/data \
  --epochs 5 \
  --batch-size 128 \
  --lr 5e-4 \
  --dim 512 \
  --amp none \
  --out data/cover_encoder.pt \
  --device cuda
```
What training does:
- Uses contrastive learning (NT-Xent) on two augmented views of each cover.
- Freezes most CLIP layers; trains the visual tower and a projection to the desired embedding dim.
- Saves weights to `data/cover_encoder.pt`, which `search` and `index` will load if present.

## Project layout
```
├── data/
│   ├── covers/                 # images (inputs)
│   ├── metadata.jsonl          # cover metadata
│   ├── covers.faiss            # built index (output)
│   ├── covers.npy              # embeddings (output)
│   ├── covers_meta.json        # aligned metadata (output)
│   └── cover_encoder.pt        # optional fine-tuned model
├── samples/                    # example photos
├── src/
│   ├── build_index.py          # CLI to build the index
│   ├── search_cover.py         # CLI to search a photo
│   ├── config.py               # central configuration and logging
│   ├── indexing/build.py       # embedding + index builder
│   ├── models/clip_model.py    # CLIP wrapper and loader
│   ├── models/index.py         # FAISS utilities (load/save/build)
│   ├── search/search.py        # search + rectification + rerank
│   ├── training/train.py       # training loop
│   ├── training/dataset.py     # dataset + augmentations
│   └── training/losses.py      # NT-Xent loss
├── export_crop.png             # latest crop from search (debug)
├── export_debug.png            # latest detection overlay (debug)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Tips and troubleshooting
- If index dimension does not match model, rebuild the index after changing the encoder or its projection.
- If covers are not found, set `COVERS_ROOT` or ensure a `covers/` directory exists under your project/data root.
- CUDA out-of-memory during training: reduce `--batch-size`, use `--amp fp16`, or train on CPU.
- FAISS GPU is optional; the code writes a CPU index that is portable.

## License
Apache 2.0. See `LICENSE` for details.
