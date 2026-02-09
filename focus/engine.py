"""Processing engine with placeholder models and parallel pipeline.

Architecture overview
---------------------
The ``Engine`` orchestrates three parallel processing branches that run
concurrently via ``concurrent.futures.ThreadPoolExecutor``:

1. **Visual branch** - face detection, AU, gaze, head-pose, blink
2. **Audio branch** - VAD, active-speaker detection, prosody
3. **Text branch** - Whisper STT, word counting

Each branch exposes a thin function signature documented below.
Replace the ``_mock_*`` helpers with real model calls (e.g. from exordium)
to move from demo mode to production.

Frame-skipping & buffering
--------------------------
* ``target_fps`` controls how many frames per second are actually sent
  through the heavy model path.  Intermediate frames are skipped and
  the last known result is reused so the UI never stalls.
* A ``FrameBuffer`` (ring-buffer) decouples capture from inference.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FaceDetectionResult:
    """Output of face detector.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coords.
              Shape: (4,), dtype: int
        confidence: Detection confidence in [0, 1].
    """

    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0


@dataclass
class LandmarkResult:
    """68-point facial landmarks.

    Attributes:
        points: Array of shape (68, 2), dtype float32, pixel coordinates.
    """

    points: np.ndarray = field(default_factory=lambda: np.zeros((68, 2), dtype=np.float32))


@dataclass
class GazeResult:
    """Per-eye gaze direction from L2CS-Net.

    Attributes:
        pitch: Vertical gaze angle in degrees. Shape: scalar float.
        yaw: Horizontal gaze angle in degrees. Shape: scalar float.
        looking_at_camera: Whether gaze falls within the camera threshold.
    """

    pitch: float = 0.0
    yaw: float = 0.0
    looking_at_camera: bool = False


@dataclass
class HeadPoseResult:
    """Head orientation (Euler angles).

    Attributes:
        yaw: Rotation around vertical axis in degrees (left/right).
        pitch: Rotation around lateral axis in degrees (up/down).
        roll: Rotation around forward axis in degrees (tilt).
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class ActionUnitResult:
    """OpenGraphAU action-unit activations.

    Attributes:
        activations: Dict mapping AU name (e.g. "AU12") to intensity [0, 5].
        dominant_au: The AU with the highest activation.
    """

    activations: dict[str, float] = field(default_factory=dict)
    dominant_au: str = "N/A"


@dataclass
class BlinkResult:
    """Blink detection output.

    Attributes:
        is_blinking: True if a blink is detected in this frame.
        blink_rate: Estimated blinks per minute (rolling average).
        ear: Eye aspect ratio. Shape: scalar float.
    """

    is_blinking: bool = False
    blink_rate: float = 0.0
    ear: float = 0.3


@dataclass
class AudioResult:
    """Audio analysis output (VAD + prosody).

    Attributes:
        is_speaking: True if voice activity detected.
        speaking_ratio: Fraction of recent window spent speaking [0, 1].
        prosody_engagement: Prosody-derived engagement indicator [0, 1].
    """

    is_speaking: bool = False
    speaking_ratio: float = 0.0
    prosody_engagement: float = 0.5


@dataclass
class TextResult:
    """Speech-to-text output.

    Attributes:
        transcript: Latest transcribed text segment.
        word_count: Cumulative word count during active speaking.
    """

    transcript: str = ""
    word_count: int = 0


@dataclass
class EngineResult:
    """Aggregated output of a single processing cycle."""

    face: FaceDetectionResult = field(default_factory=FaceDetectionResult)
    landmarks: LandmarkResult = field(default_factory=LandmarkResult)
    gaze: GazeResult = field(default_factory=GazeResult)
    head_pose: HeadPoseResult = field(default_factory=HeadPoseResult)
    action_units: ActionUnitResult = field(default_factory=ActionUnitResult)
    blink: BlinkResult = field(default_factory=BlinkResult)
    audio: AudioResult = field(default_factory=AudioResult)
    text: TextResult = field(default_factory=TextResult)
    engagement_score: float = 50.0
    eye_contact_pct: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Placeholder model functions — replace with real implementations
# ---------------------------------------------------------------------------


def detect_face(frame: np.ndarray) -> FaceDetectionResult:
    """Detect the primary face in a BGR frame.

    TODO: Replace with a real detector (e.g. RetinaFace, SCRFD from exordium).

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.

    Returns:
        FaceDetectionResult with bbox and confidence.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = min(h, w) // 3
    return FaceDetectionResult(
        bbox=(cx - size, cy - size, cx + size, cy + size),
        confidence=0.95,
    )


def detect_landmarks(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> LandmarkResult:
    """Detect 68 facial landmarks inside the given bounding box.

    TODO: Replace with a real landmark detector (e.g. FAN, MediaPipe from exordium).

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        bbox: Face bounding box (x1, y1, x2, y2).

    Returns:
        LandmarkResult with 68 landmark points, shape (68, 2).
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    points = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        angle = 2 * math.pi * i / 68
        r = 0.35 * min(bw, bh)
        points[i] = [cx + r * math.cos(angle), cy + r * math.sin(angle)]
    return LandmarkResult(points=points)


def estimate_gaze(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> GazeResult:
    """Estimate gaze direction using L2CS-Net.

    TODO: Replace with real L2CS-Net inference from exordium.

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        bbox: Face bounding box (x1, y1, x2, y2).

    Returns:
        GazeResult with pitch, yaw (degrees) and looking_at_camera flag.
              Threshold for looking_at_camera: |pitch| < 10 and |yaw| < 10.
    """
    t = time.time()
    pitch = 8.0 * math.sin(t * 0.5)
    yaw = 6.0 * math.cos(t * 0.7)
    return GazeResult(
        pitch=pitch,
        yaw=yaw,
        looking_at_camera=abs(pitch) < 10 and abs(yaw) < 10,
    )


def estimate_head_pose(frame: np.ndarray, landmarks: np.ndarray) -> HeadPoseResult:
    """Estimate head pose from facial landmarks.

    TODO: Replace with real head-pose estimator (e.g. SolvePnP or 6DRepNet from exordium).

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        landmarks: Facial landmark array, shape (68, 2), dtype float32.

    Returns:
        HeadPoseResult with yaw, pitch, roll in degrees.
    """
    t = time.time()
    return HeadPoseResult(
        yaw=12.0 * math.sin(t * 0.3),
        pitch=8.0 * math.cos(t * 0.4),
        roll=5.0 * math.sin(t * 0.6),
    )


def detect_action_units(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> ActionUnitResult:
    """Detect facial action units using OpenGraphAU.

    TODO: Replace with real OpenGraphAU inference from exordium.

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        bbox: Face bounding box (x1, y1, x2, y2).

    Returns:
        ActionUnitResult with AU activations dict and dominant AU.
              Keys: "AU1", "AU2", ..., "AU26". Values: intensity [0, 5].
    """
    t = time.time()
    aus = {
        "AU1": 1.0 + math.sin(t * 0.5),
        "AU2": 0.8 + math.cos(t * 0.6),
        "AU4": 0.5 + 0.5 * math.sin(t * 0.3),
        "AU6": 2.0 + math.sin(t * 0.4),
        "AU7": 1.5 + math.cos(t * 0.35),
        "AU10": 0.7 + 0.3 * math.sin(t * 0.45),
        "AU12": 2.5 + math.sin(t * 0.25),
        "AU14": 0.4 + 0.4 * math.cos(t * 0.55),
        "AU15": 0.3 + 0.3 * math.sin(t * 0.65),
        "AU17": 0.6 + 0.6 * math.cos(t * 0.2),
        "AU23": 0.2 + 0.2 * math.sin(t * 0.75),
        "AU24": 0.5 + 0.5 * math.cos(t * 0.85),
        "AU25": 1.0 + math.sin(t * 0.15),
        "AU26": 0.8 + 0.8 * math.cos(t * 0.95),
    }
    dominant = max(aus, key=aus.get)  # type: ignore[arg-type]
    return ActionUnitResult(activations=aus, dominant_au=dominant)


def detect_blink(frame: np.ndarray, landmarks: np.ndarray) -> BlinkResult:
    """Detect blinks via eye-aspect-ratio (EAR) thresholding.

    TODO: Replace with real blink detector from exordium.

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        landmarks: Facial landmark array, shape (68, 2), dtype float32.

    Returns:
        BlinkResult with is_blinking flag, blink_rate (blinks/min), and EAR value.
              EAR threshold for blink: < 0.21.
    """
    t = time.time()
    ear = 0.28 + 0.08 * math.sin(t * 2.0)
    is_blinking = ear < 0.21
    return BlinkResult(is_blinking=is_blinking, blink_rate=14.0 + 2.0 * math.sin(t * 0.1), ear=ear)


def process_audio(audio_chunk: np.ndarray | None = None) -> AudioResult:
    """Process an audio chunk for VAD, active-speaker detection, and prosody.

    TODO: Replace with real audio pipeline from exordium:
          - VAD (e.g. Silero VAD) for turn-taking
          - Active speaker detection
          - Prosody / F0 estimation for engagement

    Args:
        audio_chunk: Audio waveform, shape (N,), dtype float32, sample rate 16kHz.
                     None means no audio available this cycle.

    Returns:
        AudioResult with is_speaking, speaking_ratio, prosody_engagement.
    """
    t = time.time()
    is_speaking = math.sin(t * 0.2) > 0.0
    return AudioResult(
        is_speaking=is_speaking,
        speaking_ratio=0.4 + 0.2 * math.sin(t * 0.15),
        prosody_engagement=0.5 + 0.3 * math.cos(t * 0.25),
    )


def transcribe_speech(audio_chunk: np.ndarray | None = None) -> TextResult:
    """Transcribe speech using Whisper and count words.

    TODO: Replace with real Whisper STT from exordium.

    Args:
        audio_chunk: Audio waveform, shape (N,), dtype float32, sample rate 16kHz.
                     None means no audio available this cycle.

    Returns:
        TextResult with transcript string and cumulative word_count.
    """
    return TextResult(transcript="", word_count=0)


def compute_engagement(
    gaze: GazeResult,
    head_pose: HeadPoseResult,
    action_units: ActionUnitResult,
    audio: AudioResult,
    blink: BlinkResult,
) -> float:
    """Compute overall engagement score from all modalities.

    TODO: Replace with a trained engagement model or rule-based fusion.

    Current mock formula (weighted combination):
        - Eye contact (gaze near center) contributes 25%
        - Head pose (facing forward) contributes 20%
        - Action unit activation (AU6+AU12 = smile) contributes 20%
        - Audio prosody engagement contributes 20%
        - Blink rate (moderate = engaged) contributes 15%

    Args:
        gaze: GazeResult from L2CS-Net.
        head_pose: HeadPoseResult from head-pose estimator.
        action_units: ActionUnitResult from OpenGraphAU.
        audio: AudioResult from audio pipeline.
        blink: BlinkResult from blink detector.

    Returns:
        Engagement score in [0, 100].
    """
    gaze_score = max(0, 100 - (abs(gaze.pitch) + abs(gaze.yaw)) * 3)
    pose_score = max(0, 100 - (abs(head_pose.yaw) + abs(head_pose.pitch)) * 2.5)

    aus = action_units.activations
    smile = (aus.get("AU6", 0) + aus.get("AU12", 0)) / 2
    au_score = min(100, smile * 20)

    audio_score = audio.prosody_engagement * 100

    ideal_blink_rate = 15.0
    blink_score = max(0, 100 - abs(blink.blink_rate - ideal_blink_rate) * 5)

    score = (
        0.25 * gaze_score
        + 0.20 * pose_score
        + 0.20 * au_score
        + 0.20 * audio_score
        + 0.15 * blink_score
    )
    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# Engine — orchestrates parallel processing
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Thread-safe ring buffer for decoupling capture from inference.

    Attributes:
        maxlen: Maximum number of frames to buffer.
    """

    def __init__(self, maxlen: int = 4) -> None:
        import threading

        self._buf: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._maxlen = maxlen

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            if len(self._buf) >= self._maxlen:
                self._buf.pop(0)
            self._buf.append(frame)

    def pop(self) -> np.ndarray | None:
        with self._lock:
            return self._buf.pop(0) if self._buf else None

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buf)


class Engine:
    """Main processing engine with parallel branch execution and frame skipping.

    Args:
        target_fps: Maximum frames per second to send through inference.
                    Frames arriving faster than this are skipped.
        max_workers: Thread pool size for parallel branch execution.
    """

    def __init__(self, target_fps: float = 15.0, max_workers: int = 3) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._target_fps = target_fps
        self._min_interval = 1.0 / target_fps
        self._last_process_time = 0.0
        self._last_result: EngineResult | None = None
        self.frame_buffer = FrameBuffer(maxlen=4)

    def should_process(self) -> bool:
        """Return True if enough time has elapsed since last inference."""
        return (time.time() - self._last_process_time) >= self._min_interval

    def process_frame(self, frame: np.ndarray) -> EngineResult:
        """Run the full processing pipeline on a single BGR frame.

        If called faster than ``target_fps``, returns the cached last result
        (frame-skipping).  Otherwise, runs visual, audio, and text branches
        in parallel using the thread pool.

        Args:
            frame: BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            EngineResult with all modality outputs and engagement score.
        """
        now = time.time()

        if not self.should_process() and self._last_result is not None:
            return self._last_result

        self._last_process_time = now

        # Step 1: face detection (required by downstream)
        face = detect_face(frame)

        # Step 2: parallel branches that depend on face bbox
        landmarks_future = self._pool.submit(detect_landmarks, frame, face.bbox)
        gaze_future = self._pool.submit(estimate_gaze, frame, face.bbox)
        au_future = self._pool.submit(detect_action_units, frame, face.bbox)
        audio_future = self._pool.submit(process_audio, None)
        text_future = self._pool.submit(transcribe_speech, None)

        landmarks = landmarks_future.result()
        gaze = gaze_future.result()
        aus = au_future.result()
        audio = audio_future.result()
        text = text_future.result()

        # Step 3: head pose and blink depend on landmarks
        head_pose_future = self._pool.submit(estimate_head_pose, frame, landmarks.points)
        blink_future = self._pool.submit(detect_blink, frame, landmarks.points)

        head_pose = head_pose_future.result()
        blink = blink_future.result()

        # Step 4: engagement fusion
        engagement = compute_engagement(gaze, head_pose, aus, audio, blink)

        if gaze.looking_at_camera:
            eye_contact = 100.0
        else:
            eye_contact = max(0, 100 - (abs(gaze.pitch) + abs(gaze.yaw)) * 5)

        result = EngineResult(
            face=face,
            landmarks=landmarks,
            gaze=gaze,
            head_pose=head_pose,
            action_units=aus,
            blink=blink,
            audio=audio,
            text=text,
            engagement_score=engagement,
            eye_contact_pct=eye_contact,
            timestamp=now,
        )
        self._last_result = result
        return result

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
