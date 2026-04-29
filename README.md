# SatDamage

A deep learning system for classifying building damage severity from satellite imagery after natural disasters. Given paired pre- and post-disaster satellite images with building annotations, SatDamage predicts per-building damage levels: **no-damage**, **minor-damage**, **major-damage**, or **destroyed**.

Built on the [xView2 (xBD)](https://xview2.org/) dataset, covering hurricanes, earthquakes, volcanoes, tsunamis, and wildfires.

## Demo

Experience the model in action on our interactive web interface:
**[SatDamage Live Demo](https://satdamage-demo.streamlit.app/)**


## How It Works

For each annotated building, the pipeline:
1. Crops the building from both the pre- and post-disaster images
2. Concatenates the crops into a 6-channel (128x128) tensor
3. Classifies damage severity using a trained CNN or EfficientNet model

The system supports both **binary** (damaged / not damaged) and **multiclass** (4-class) classification.

## Project Structure

```
satdamage/
  params.py              # Configuration & hyperparameters
  utils.py               # Visualization utilities
  ml_logic/
    model.py             # Model architectures & training loops
    preprocessor.py      # Data loading, crop extraction, tf.data pipeline
    registry.py          # Model save/load (local & GCS)
  interface/
    main.py              # Training entry point
    evaluate_light.py    # Evaluation-only entry point
  api/
    fast.py              # FastAPI inference endpoints
```

## Model Architectures

| Architecture | Description |
|---|---|
| `efficientnet` (default) | EfficientNetV2B0 with 6-to-3 channel projection, 2-phase training (warmup + fine-tune) |
| `cnn_concat` | Simple 4-block CNN on concatenated 6-channel input |
| `cnn_dual` | Dual-stream siamese encoder with squeeze-and-excitation attention |

Select the architecture in `params.py` via `MODEL_ARCHITECTURE`.

## Setup

### Requirements

- Python 3.10+
- TensorFlow 2.16.2

### Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root:

```bash
# Required
DATA_DIR=/path/to/xview2/dataset
MODEL_TARGET=local              # "local" or "gcs"

# Optional
CROPS_DIR=data/crops            # Where extracted crops are cached
MODEL_FILENAME=model.keras      # For evaluate_light

# GCS (required if MODEL_TARGET=gcs)
GCP_PROJECT=your-project
GCP_REGION=your-region
BUCKET_NAME=your-bucket
```

## Usage

### Training

```bash
python -m satdamage.interface.main
```

This will:
1. Scan the xView2 dataset for image pairs
2. Extract and cache building crops to disk
3. Split into train/val/test (70/15/15)
4. Train the selected model architecture
5. Evaluate on the test set and save metrics

### Evaluation Only

Run a pre-trained model on the test set without retraining:

```bash
python -m satdamage.interface.evaluate_light
```

### API

Start the inference API:

```bash
uvicorn satdamage.api.fast:app --reload
```

**Endpoints:**

- `POST /predict` — Upload pre/post images + annotation JSONs, get per-building damage predictions
- `GET /models` — List loaded models
- `GET /` — Health check

### Docker

```bash
docker build -t satdamage .
docker run -p 8000:8000 -e PORT=8000 satdamage
```

## Dataset Format

SatDamage expects the [xBD dataset](https://xview2.org/) structure:

```
xview2_root/
  {event_name}/
    images/
      {event_name}_00000000_pre_disaster.png
      {event_name}_00000000_post_disaster.png
    labels/
      {event_name}_00000000_pre_disaster.json
      {event_name}_00000000_post_disaster.json
```

Labels are GeoJSON files containing building polygons with damage severity annotations.

## License

MIT
