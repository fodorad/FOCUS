"""FOCUS Gradio demo application.

Launch with:
    python -m focus.app
    # or after installing:
    focus
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

from focus.engine import Engine, EngineResult
from focus.state import MetricSnapshot, SessionState, engagement_label
from focus.visualization import draw_overlays

if TYPE_CHECKING:
    import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Global engine & state (shared across Gradio callbacks)
# ---------------------------------------------------------------------------
engine = Engine(target_fps=15.0, max_workers=3)
session = SessionState(window=30.0)


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------


def build_engagement_html(score: float) -> str:
    """Render engagement score as a styled progress bar with threshold label."""
    label, color = engagement_label(score)
    pct = max(0, min(100, score))
    return f"""
    <div style="padding: 12px 16px; font-family: sans-serif;">
      <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 6px;">
        <span style="font-size: 1.4em; font-weight: 700; min-width: 120px;">
          Engagement: {pct:.0f}
        </span>
        <div style="flex: 1; background: #2a2a2a; border-radius: 8px; height: 28px;
                    overflow: hidden;">
          <div style="width: {pct}%; height: 100%; background: {color};
                      border-radius: 8px; transition: width 0.3s ease;"></div>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.1em; font-weight: 600; color: {color};">
          {label}
        </span>
        {_threshold_dots(score)}
      </div>
    </div>
    """


def _threshold_dots(score: float) -> str:
    from focus.state import ENGAGEMENT_THRESHOLDS

    _, active_color = engagement_label(score)
    dots = []
    for _threshold, lbl, col in ENGAGEMENT_THRESHOLDS:
        opacity = "1.0" if col == active_color else "0.3"
        dots.append(
            f'<span style="display:inline-block; width:12px; height:12px; '
            f"border-radius:50%; background:{col}; opacity:{opacity}; "
            f'margin: 0 2px;" title="{lbl}"></span>'
        )
    return "".join(dots)


def build_metrics_html(result: EngineResult, fps: float) -> str:
    """Render the right-side metrics panel."""
    aus = result.action_units.activations
    top_aus = sorted(aus.items(), key=lambda kv: kv[1], reverse=True)[:5]
    au_rows = "".join(
        f'<tr><td style="padding:2px 8px;">{name}</td>'
        f'<td style="padding:2px 8px;">{val:.1f}</td></tr>'
        for name, val in top_aus
    )

    looking = "Yes" if result.gaze.looking_at_camera else "No"
    looking_color = "#2ecc71" if result.gaze.looking_at_camera else "#e74c3c"

    return f"""
    <div style="padding: 10px; font-family: monospace; font-size: 0.9em; line-height: 1.7;">
      <div style="margin-bottom: 10px;">
        <b>Eye Contact:</b>
        <span style="color: {looking_color}; font-weight:bold;">{looking}</span>
        ({result.eye_contact_pct:.0f}%)
      </div>
      <div style="margin-bottom: 10px;">
        <b>Blink Rate:</b> {result.blink.blink_rate:.1f} /min
        &nbsp; EAR: {result.blink.ear:.2f}
      </div>
      <div style="margin-bottom: 10px;">
        <b>Head Pose:</b><br/>
        &nbsp; Yaw: {result.head_pose.yaw:+.1f}&deg;
        &nbsp; Pitch: {result.head_pose.pitch:+.1f}&deg;
        &nbsp; Roll: {result.head_pose.roll:+.1f}&deg;
      </div>
      <div style="margin-bottom: 10px;">
        <b>Speaking:</b> {result.audio.speaking_ratio:.0%}
        &nbsp; ({("Active" if result.audio.is_speaking else "Silent")})
      </div>
      <div style="margin-bottom: 10px;">
        <b>Dominant AU:</b> {result.action_units.dominant_au}
      </div>
      <div style="margin-bottom: 4px;"><b>Top AUs:</b></div>
      <table style="border-collapse:collapse; font-size:0.85em;">
        {au_rows}
      </table>
      <div style="margin-top: 10px; color: #888; font-size: 0.8em;">
        FPS: {fps:.1f}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------


def build_timeline_plot() -> plt.Figure:
    """Create matplotlib figure with 3 subplots sharing the x-axis."""
    timeline = session.get_timeline()

    fig, axes = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True, dpi=80)
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(hspace=0.15, left=0.08, right=0.97, top=0.95, bottom=0.10)

    ts = timeline.timestamps

    # Engagement
    ax0 = axes[0]
    ax0.set_facecolor("#16213e")
    if ts:
        ax0.plot(ts, timeline.engagement, color="#3498db", linewidth=1.5)
        ax0.fill_between(ts, timeline.engagement, alpha=0.15, color="#3498db")
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Engage", fontsize=8, color="#ccc")
    ax0.tick_params(colors="#888", labelsize=7)
    ax0.grid(True, alpha=0.15)

    # Eye contact
    ax1 = axes[1]
    ax1.set_facecolor("#16213e")
    if ts:
        ax1.plot(ts, timeline.eye_contact, color="#2ecc71", linewidth=1.5)
        ax1.fill_between(ts, timeline.eye_contact, alpha=0.15, color="#2ecc71")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Eye %", fontsize=8, color="#ccc")
    ax1.tick_params(colors="#888", labelsize=7)
    ax1.grid(True, alpha=0.15)

    # Blink events
    ax2 = axes[2]
    ax2.set_facecolor("#16213e")
    if ts:
        ax2.fill_between(ts, timeline.blink_events, step="mid", alpha=0.5, color="#e74c3c")
        ax2.plot(ts, timeline.blink_events, color="#e74c3c", linewidth=1, drawstyle="steps-mid")
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_ylabel("Blink", fontsize=8, color="#ccc")
    ax2.set_xlabel("Time (s)", fontsize=8, color="#ccc")
    ax2.tick_params(colors="#888", labelsize=7)
    ax2.grid(True, alpha=0.15)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("#333")

    return fig


# ---------------------------------------------------------------------------
# Processing callback
# ---------------------------------------------------------------------------


def _make_default_result() -> EngineResult:
    return EngineResult(timestamp=time.time())


def process_webcam_frame(
    frame: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str, plt.Figure]:
    """Process a single webcam frame and return all UI updates.

    Returns:
        (raw_frame, processed_frame, engagement_html, metrics_html, timeline_fig)
    """
    if frame is None:
        result = _make_default_result()
        fig = build_timeline_plot()
        return None, None, build_engagement_html(0), build_metrics_html(result, 0), fig

    result = engine.process_frame(frame)

    snapshot = MetricSnapshot(
        timestamp=result.timestamp,
        engagement_score=result.engagement_score,
        eye_contact_pct=result.eye_contact_pct,
        blink_event=result.blink.is_blinking,
        blink_rate=result.blink.blink_rate,
        head_yaw=result.head_pose.yaw,
        head_pitch=result.head_pose.pitch,
        head_roll=result.head_pose.roll,
        speaking_ratio=result.audio.speaking_ratio,
        dominant_au=result.action_units.dominant_au,
    )
    session.push(snapshot)

    processed = draw_overlays(frame, result)

    # Convert BGR to RGB for Gradio display
    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    engagement_html = build_engagement_html(result.engagement_score)
    metrics_html = build_metrics_html(result, session.fps)
    fig = build_timeline_plot()

    return raw_rgb, proc_rgb, engagement_html, metrics_html, fig


def process_video_file(video_path: str):
    """Process an uploaded video file frame-by-frame as a generator.

    Yields:
        (raw_frame, processed_frame, engagement_html, metrics_html, timeline_fig)
    """
    if not video_path:
        return

    session.reset()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / min(video_fps, engine._target_fps)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            result = engine.process_frame(frame)

            snapshot = MetricSnapshot(
                timestamp=result.timestamp,
                engagement_score=result.engagement_score,
                eye_contact_pct=result.eye_contact_pct,
                blink_event=result.blink.is_blinking,
                blink_rate=result.blink.blink_rate,
                head_yaw=result.head_pose.yaw,
                head_pitch=result.head_pose.pitch,
                head_roll=result.head_pose.roll,
                speaking_ratio=result.audio.speaking_ratio,
                dominant_au=result.action_units.dominant_au,
            )
            session.push(snapshot)

            processed = draw_overlays(frame, result)
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            engagement_html = build_engagement_html(result.engagement_score)
            metrics_html = build_metrics_html(result, session.fps)
            fig = build_timeline_plot()

            yield raw_rgb, proc_rgb, engagement_html, metrics_html, fig

            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    css = """
    .focus-title {
        text-align: center;
        font-size: 1.5em;
        font-weight: 700;
        padding: 8px 0;
        color: #e0e0e0;
    }
    .engagement-bar { min-height: 80px; }
    .metrics-panel { min-height: 300px; }
    .timeline-area { min-height: 200px; }
    """

    with gr.Blocks(
        title="FOCUS - Engagement Analysis",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=css,
    ) as app:
        # Title
        gr.HTML(
            '<div class="focus-title">'
            "FOCUS: Fast Observation and Character Understanding System"
            "</div>"
        )

        # ── Row 1: Engagement score bar ──────────────────────────────
        with gr.Row():
            engagement_display = gr.HTML(
                value=build_engagement_html(50),
                elem_classes=["engagement-bar"],
            )

        # ── Row 2: Camera feeds + Metrics ────────────────────────────
        with gr.Row():
            # Column 1 — raw camera (15%)
            with gr.Column(scale=3, min_width=120):
                gr.Markdown("**Raw Feed**")
                raw_image = gr.Image(
                    label="Camera",
                    height=240,
                    show_label=False,
                )

                # Input controls
                gr.Markdown("---")
                gr.Markdown("**Input Source**")
                with gr.Tab("Webcam"):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam",
                        height=1,
                        visible=False,
                    )
                    gr.Markdown("*Enable webcam above to start*")
                with gr.Tab("Video File"):
                    video_input = gr.Video(label="Upload Video")
                    video_btn = gr.Button("Process Video", variant="primary")

            # Column 2 — processed feed (50%)
            with gr.Column(scale=10, min_width=300):
                gr.Markdown("**Processed Feed**")
                processed_image = gr.Image(
                    label="Processed",
                    height=480,
                    show_label=False,
                )

            # Column 3 — metrics panel (35%)
            with gr.Column(scale=7, min_width=200):
                gr.Markdown("**Metrics**")
                metrics_display = gr.HTML(
                    value=build_metrics_html(_make_default_result(), 0),
                    elem_classes=["metrics-panel"],
                )

        # ── Row 3: Timeline ──────────────────────────────────────────
        with gr.Row():
            timeline_plot = gr.Plot(
                value=build_timeline_plot(),
                label="Timeline (last 30s)",
                elem_classes=["timeline-area"],
            )

        # ── Webcam streaming callback ────────────────────────────────
        webcam_input.stream(
            fn=process_webcam_frame,
            inputs=[webcam_input],
            outputs=[
                raw_image,
                processed_image,
                engagement_display,
                metrics_display,
                timeline_plot,
            ],
        )

        # ── Video file callback ──────────────────────────────────────
        video_btn.click(
            fn=process_video_file,
            inputs=[video_input],
            outputs=[
                raw_image,
                processed_image,
                engagement_display,
                metrics_display,
                timeline_plot,
            ],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the FOCUS demo."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
