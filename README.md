# FOCUS
FOCUS: Fast Observation and Character Understanding System

[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

[![CI](https://github.com/fodorad/FOCUS/workflows/CI/badge.svg)](https://github.com/fodorad/FOCUS/actions)
[![Coverage](https://codecov.io/gh/fodorad/FOCUS/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/FOCUS)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Real-time multimodal engagement analysis demo built with Gradio. Combines visual (AU detection, gaze tracking, head pose), audio (VAD, prosody), and textual (STT, word count) processing to compute a live engagement score.

## Setup

**Requirements:** Python 3.12+, a webcam (for live mode)

```bash
# Clone the repository
git clone https://github.com/fodorad/FOCUS.git
cd FOCUS

# Install in editable mode (recommended for development)
pip install -e ".[dev]"

# Or install directly
pip install .
```

## Usage

### Launch the demo

```bash
# Using the CLI entry point
focus

# Or run as a module
python -m focus.app
```

The Gradio interface opens at `http://localhost:7860`.

### Demo modes

- **Webcam mode** — enable your webcam in the interface for real-time engagement analysis
- **Video file mode** — upload a video file and click "Process Video" for offline analysis

### UI layout

| Row | Content |
|-----|---------|
| **Top** | Engagement score (0–100) progress bar with threshold labels (Very Low / Low / Neutral / High / Very High) |
| **Middle** | Raw camera feed (15%) &#124; Processed feed with overlays (50%) &#124; Metrics panel (35%) |
| **Bottom** | 30-second rolling timeline: engagement, eye contact, blink events |

## Architecture

```
focus/
├── __init__.py         # Package metadata and paths
├── app.py              # Gradio UI layout and callbacks
├── engine.py           # Parallel processing pipeline (placeholder models)
├── state.py            # Thread-safe rolling metric buffers
└── visualization.py    # OpenCV frame overlay drawing
```

### Processing pipeline

The engine runs three parallel branches via `ThreadPoolExecutor`:

1. **Visual** — face detection, landmarks, gaze (L2CS-Net), head pose, action units (OpenGraphAU), blink detection
2. **Audio** — VAD, active speaker detection, prosody estimation
3. **Text** — Whisper STT, word counting

Frame skipping keeps the UI responsive: frames arriving faster than `target_fps` reuse the last inference result.

### Placeholder functions

All model functions in `engine.py` are documented with expected input/output types and shapes. Replace them with real implementations (e.g. from [exordium](https://github.com/fodorad/exordium)):

| Function | Input | Output |
|----------|-------|--------|
| `detect_face` | BGR frame `(H,W,3) uint8` | `FaceDetectionResult` (bbox, confidence) |
| `detect_landmarks` | frame + bbox | `LandmarkResult` (68,2) float32 |
| `estimate_gaze` | frame + bbox | `GazeResult` (pitch, yaw, looking_at_camera) |
| `estimate_head_pose` | frame + landmarks | `HeadPoseResult` (yaw, pitch, roll) |
| `detect_action_units` | frame + bbox | `ActionUnitResult` (AU dict, dominant AU) |
| `detect_blink` | frame + landmarks | `BlinkResult` (is_blinking, rate, EAR) |
| `process_audio` | audio chunk `(N,) float32 16kHz` | `AudioResult` (VAD, ratio, prosody) |
| `transcribe_speech` | audio chunk `(N,) float32 16kHz` | `TextResult` (transcript, word_count) |
| `compute_engagement` | all modality results | `float` in [0, 100] |

## Development

```bash
# Format code
make format

# Lint
make lint

# Run tests
make test
```

## Acknowledgement
- [exordium](https://github.com/fodorad/exordium) — multimodal feature extraction toolkit
- [Gradio](https://gradio.app/) — web UI framework
- [OpenCV](https://opencv.org/) — computer vision library

## Contact
* Adam Fodor (fodorad201@gmail.com)
