"""Gradio UI for pairwise taste training.

Thin UI layer that delegates to the existing training backend:
- sommelier/training/session.py   -- TrainingSession
- sommelier/training/sampler.py   -- ComparisonSampler
- sommelier/training/synthesizer.py -- ProfileSynthesizer
- sommelier/features/burst_detector.py -- BurstDetector
- sommelier/core/file_utils.py    -- FileTypeRegistry, ImageUtils
- sommelier/core/provider_factory.py -- create_ai_client
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from sommelier.core.config import load_config, BurstDetectionConfig
from sommelier.core.file_utils import FileTypeRegistry, ImageUtils
from sommelier.core.profiles import ProfileManager
from sommelier.features.burst_detector import BurstDetector
from sommelier.training.session import TrainingSession
from sommelier.training.sampler import ComparisonSampler

# ── Defaults ────────────────────────────────────────────────────────

CONFIG_PATH = Path("config.yaml")
MIN_LABELS = 15


def _load_config():
    if CONFIG_PATH.exists():
        return load_config(CONFIG_PATH)
    from sommelier.core.config import Config
    return Config()


# ── Burst detection (shared with old MCP handler logic) ─────────────

def _detect_bursts_temporal_only(
    images: list[Path],
) -> tuple[list[list[str]], list[str]]:
    """Detect bursts using only EXIF timestamps."""
    config = BurstDetectionConfig()
    detector = BurstDetector(config)
    photo_times = detector._extract_timestamps(images)

    if not photo_times:
        return [], [str(p) for p in images]

    temporal_groups = detector._create_temporal_groups(photo_times)

    bursts: list[list[str]] = []
    singletons: list[str] = []
    for group in temporal_groups:
        paths = [str(item[1]) for item in group]
        if len(paths) >= 2:
            bursts.append(paths)
        else:
            singletons.extend(paths)

    grouped_indices = {item[0] for group in temporal_groups for item in group}
    for i, img in enumerate(images):
        if i not in grouped_indices:
            singletons.append(str(img))

    return bursts, singletons


def _detect_bursts_with_embeddings(
    images: list[Path],
) -> tuple[list[list[str]], list[str]]:
    """Detect bursts using CLIP embeddings + temporal proximity."""
    config = _load_config()
    try:
        from sommelier.features.embeddings import EmbeddingExtractor
        from sommelier.core.cache import CacheManager

        cache_manager = CacheManager(
            config.paths.cache_root,
            ttl_days=config.caching.ttl_days,
            enabled=config.caching.enabled,
        )
        extractor = EmbeddingExtractor(
            config.model, config.performance, cache_manager
        )
        embeddings = extractor.extract_embeddings_batch(images, show_progress=True)

        burst_config = BurstDetectionConfig()
        detector = BurstDetector(burst_config)
        burst_groups = detector.detect_bursts(images, embeddings)

        bursts: list[list[str]] = []
        singletons: list[str] = []
        for group in burst_groups:
            paths = [str(p) for p in group]
            if len(paths) >= 2:
                bursts.append(paths)
            else:
                singletons.extend(paths)
        return bursts, singletons

    except Exception as e:
        print(f"Embedding burst detection failed ({e}), using temporal-only", file=sys.stderr)
        return _detect_bursts_temporal_only(images)


# ── Stats formatting ────────────────────────────────────────────────

def _format_stats(session: TrainingSession) -> str:
    s = session.get_stats()
    ready = "Ready" if s["ready_to_synthesize"] else f"Need {MIN_LABELS - s['total_labeled']} more"
    lines = [
        f"**Session** `{s['session_id']}`",
        f"**Profile** `{s['profile_name']}`",
        "",
        f"Pairwise: {s['pairwise_count']}  (within {s['within_burst']}, between {s['between_burst']})",
        f"Gallery:  {s['gallery_count']}",
        f"Total labeled: {s['total_labeled']}",
        f"Unique photos: {s['unique_photos_labeled']} / {s['total_photos']}",
        f"Coverage: {s['coverage_pct']}%",
        "",
        f"Synthesize: **{ready}**",
    ]
    return "\n".join(lines)


# ── Gradio app ──────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    # Mutable state shared across callbacks (single-user Gradio app)
    state: dict = {
        "session": None,
        "sampler": None,
        "comparison": None,  # current comparison dict from sampler
    }

    # ── Start training ──────────────────────────────────────────────

    def start_training(folder_path: str, profile_name: str, skip_embeddings: bool):
        folder = Path(folder_path)
        if not folder.is_dir():
            return (
                gr.update(visible=True),   # setup still visible
                gr.update(visible=False),  # training hidden
                f"Folder not found: {folder}",
                "",  # stats
            )

        images = FileTypeRegistry.list_images(folder, recursive=True)
        if not images:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                f"No image files found in {folder}",
                "",
            )

        # Burst detection
        if skip_embeddings:
            bursts, singletons = _detect_bursts_temporal_only(images)
        else:
            bursts, singletons = _detect_bursts_with_embeddings(images)

        config = _load_config()
        pm = ProfileManager(config.profiles.profiles_dir)

        session = TrainingSession.create(
            profile_name=profile_name,
            folder_path=str(folder),
            bursts=bursts,
            singletons=singletons,
        )
        session.save(pm.profiles_dir)

        state["session"] = session
        state["sampler"] = ComparisonSampler(session)

        msg = (
            f"Started session `{session.session_id}` with "
            f"{session.total_photos} photos "
            f"({len(bursts)} bursts, {len(singletons)} singletons)."
        )

        return (
            gr.update(visible=False),  # hide setup
            gr.update(visible=True),   # show training
            msg,
            _format_stats(session),
        )

    # ── Advance to next comparison ──────────────────────────────────

    def _advance():
        """Get next comparison and return UI update tuple."""
        session = state["session"]
        sampler = state["sampler"]

        comparison = sampler.get_next()
        state["comparison"] = comparison

        if comparison is None:
            config = _load_config()
            pm = ProfileManager(config.profiles.profiles_dir)
            session.save(pm.profiles_dir)
            return (
                None, None,                    # images
                gr.update(visible=False),      # pairwise buttons
                gr.update(visible=False),      # gallery panel
                gr.update(visible=False, choices=[]),  # checkboxes
                gr.update(visible=False),      # gallery submit
                "All comparisons exhausted.",   # context
                _format_stats(session),         # stats
                "",                             # reason
            )

        session.comparisons_served += 1

        if comparison["type"] == "gallery":
            photos = comparison["photos"]
            gallery_images = []
            for p in photos:
                img = ImageUtils.load_and_fix_orientation(Path(p), max_size=600)
                if img:
                    gallery_images.append((img, Path(p).name))
            choices = [f"#{i+1} {Path(p).name[:30]}" for i, p in enumerate(photos)]
            return (
                None, None,
                gr.update(visible=False),
                gr.update(visible=True, value=gallery_images),
                gr.update(visible=True, choices=choices, value=[]),
                gr.update(visible=True),
                comparison.get("context", "Select keepers from this burst."),
                _format_stats(session),
                "",
            )
        else:
            img_a = ImageUtils.load_and_fix_orientation(Path(comparison["photo_a"]))
            img_b = ImageUtils.load_and_fix_orientation(Path(comparison["photo_b"]))
            return (
                img_a, img_b,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                comparison.get("context", ""),
                _format_stats(session),
                "",
            )

    # ── Submit pairwise ─────────────────────────────────────────────

    def submit_pairwise(choice: str, reason: str):
        session = state["session"]
        comp = state["comparison"]
        if not comp or comp["type"] == "gallery":
            return _advance()

        session.add_pairwise(
            photo_a=comp["photo_a"],
            photo_b=comp["photo_b"],
            choice=choice,
            reason=reason,
            comparison_type=comp["comparison_type"],
        )
        return _advance()

    # ── Submit gallery ──────────────────────────────────────────────

    def submit_gallery(selected: list[str], reason: str):
        session = state["session"]
        comp = state["comparison"]
        if not comp or comp["type"] != "gallery":
            return _advance()

        selected_indices = []
        for sel in selected:
            try:
                idx = int(sel.split("#")[1].split(" ")[0]) - 1
                selected_indices.append(idx)
            except (IndexError, ValueError):
                pass

        session.add_gallery(
            photos=comp["photos"],
            selected_indices=selected_indices,
            reason=reason,
        )
        return _advance()

    # ── Skip ────────────────────────────────────────────────────────

    def skip_current():
        return _advance()

    # ── Synthesize ──────────────────────────────────────────────────

    def synthesize_profile():
        session = state["session"]
        if session is None:
            return "No active session."

        stats = session.get_stats()
        if not stats["ready_to_synthesize"]:
            return f"Need at least {MIN_LABELS} labels (have {stats['total_labeled']})."

        config = _load_config()
        pm = ProfileManager(config.profiles.profiles_dir)

        from sommelier.core.provider_factory import create_ai_client
        from sommelier.training.synthesizer import ProfileSynthesizer

        ai_client = create_ai_client(config)
        synthesizer = ProfileSynthesizer(ai_client, pm)

        existing = None
        if pm.profile_exists(session.profile_name):
            existing = pm.load_profile(session.profile_name)

        profile = synthesizer.synthesize(session, session.profile_name, existing)

        session.status = "completed"
        session.save(pm.profiles_dir)

        return (
            f"Profile `{session.profile_name}` "
            f"{'refined' if existing else 'created'} from "
            f"{stats['total_labeled']} labels. "
            f"Saved to `{pm.profiles_dir / (session.profile_name + '.json')}`."
        )

    # ── Build UI ────────────────────────────────────────────────────

    with gr.Blocks(title="Sommelier Taste Trainer") as app:
        gr.Markdown("# Sommelier Taste Trainer")
        gr.Markdown("Train a taste profile by comparing photos side-by-side.")

        # ── Setup panel ─────────────────────────────────────────────
        with gr.Group(visible=True) as setup_panel:
            folder_input = gr.Textbox(label="Photo folder path", placeholder="C:\\Photos\\Unsorted")
            profile_input = gr.Textbox(label="Profile name", placeholder="my-photos")
            skip_emb = gr.Checkbox(label="Skip embeddings (faster, temporal-only burst detection)", value=False)
            start_btn = gr.Button("Start Training", variant="primary")
            setup_msg = gr.Markdown("")

        # ── Training panel ──────────────────────────────────────────
        with gr.Group(visible=False) as training_panel:
            with gr.Row():
                # Left: comparison area
                with gr.Column(scale=3):
                    context_display = gr.Markdown("")

                    # Pairwise images
                    with gr.Row():
                        image_a = gr.Image(label="Photo A (Left)", type="pil", height=400)
                        image_b = gr.Image(label="Photo B (Right)", type="pil", height=400)

                    # Pairwise buttons
                    with gr.Row(visible=False) as pairwise_btns:
                        left_btn = gr.Button("Left", variant="secondary", size="lg")
                        both_btn = gr.Button("Both", variant="primary", size="lg")
                        right_btn = gr.Button("Right", variant="secondary", size="lg")
                        neither_btn = gr.Button("Neither", variant="stop", size="lg")

                    # Gallery
                    gallery_display = gr.Gallery(
                        label="Burst photos",
                        columns=4, rows=5, height="auto",
                        object_fit="contain", visible=False,
                    )
                    gallery_checks = gr.CheckboxGroup(
                        label="Select keepers", choices=[], visible=False,
                    )
                    gallery_submit = gr.Button("Submit Selection", variant="primary", visible=False)

                    # Common controls
                    reason_input = gr.Textbox(
                        label="Why? (optional but helps profile quality)",
                        placeholder="e.g., better expression, sharper focus",
                        lines=2,
                    )
                    with gr.Row():
                        skip_btn = gr.Button("Skip")
                        synthesize_btn = gr.Button("Synthesize Profile", variant="primary")

                    synth_msg = gr.Markdown("")

                # Right: stats sidebar
                with gr.Column(scale=1):
                    stats_display = gr.Markdown("")

        # ── Wiring ──────────────────────────────────────────────────

        advance_outputs = [
            image_a, image_b,
            pairwise_btns, gallery_display, gallery_checks,
            gallery_submit,
            context_display, stats_display, reason_input,
        ]

        start_btn.click(
            start_training,
            inputs=[folder_input, profile_input, skip_emb],
            outputs=[setup_panel, training_panel, setup_msg, stats_display],
        ).then(
            skip_current,  # load first comparison
            outputs=advance_outputs,
        )

        left_btn.click(
            lambda r: submit_pairwise("left", r),
            inputs=[reason_input], outputs=advance_outputs,
        )
        right_btn.click(
            lambda r: submit_pairwise("right", r),
            inputs=[reason_input], outputs=advance_outputs,
        )
        both_btn.click(
            lambda r: submit_pairwise("both", r),
            inputs=[reason_input], outputs=advance_outputs,
        )
        neither_btn.click(
            lambda r: submit_pairwise("neither", r),
            inputs=[reason_input], outputs=advance_outputs,
        )

        gallery_submit.click(
            submit_gallery,
            inputs=[gallery_checks, reason_input],
            outputs=advance_outputs,
        )

        skip_btn.click(skip_current, outputs=advance_outputs)

        synthesize_btn.click(synthesize_profile, outputs=[synth_msg])

    return app


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, inbrowser=True)
