"""Gradio UI for reviewing improvement candidates."""

import csv
from pathlib import Path
from typing import List, Dict, Optional
import gradio as gr
from PIL import Image


def load_candidates_from_csv(candidates_folder: Path) -> List[Dict]:
    """
    Load improvement candidates from CSV file.

    Args:
        candidates_folder: Path to ImprovementCandidates folder.

    Returns:
        List of candidate dictionaries.
    """
    csv_path = candidates_folder / "improvement_candidates.csv"
    if not csv_path.exists():
        return []

    candidates = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(row)

    return candidates


def save_candidates_to_csv(candidates: List[Dict], candidates_folder: Path):
    """
    Save updated candidates back to CSV file.

    Args:
        candidates: List of candidate dictionaries.
        candidates_folder: Path to ImprovementCandidates folder.
    """
    csv_path = candidates_folder / "improvement_candidates.csv"

    if not candidates:
        return

    fieldnames = list(candidates[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def create_review_interface(candidates_folder: Path):
    """
    Create Gradio interface for reviewing improvement candidates.

    Args:
        candidates_folder: Path to ImprovementCandidates folder.

    Returns:
        Gradio Blocks interface.
    """
    candidates = load_candidates_from_csv(candidates_folder)

    if not candidates:
        with gr.Blocks(title="No Candidates") as demo:
            gr.Markdown("# No Improvement Candidates Found")
            gr.Markdown("No photos were flagged as improvement candidates.")
        return demo

    with gr.Blocks(title="Photo Improvement Review", theme=gr.themes.Soft()) as demo:
        # Track current state - MUST be inside Blocks context
        current_index = gr.State(value=0)
        candidates_state = gr.State(value=candidates)
        gr.Markdown("# Photo Improvement Review")
        gr.Markdown("Review photos flagged for potential AI improvement. Approve or reject each candidate.")

        with gr.Row():
            with gr.Column(scale=2):
                # Photo display
                photo_display = gr.Image(
                    label="Photo",
                    type="pil",
                    interactive=False,
                    height=500
                )

            with gr.Column(scale=1):
                # Info display
                progress_text = gr.Markdown("**Progress:** 1 / " + str(len(candidates)))
                filename_text = gr.Textbox(label="Filename", interactive=False)
                contextual_value_text = gr.Textbox(label="Contextual Value", interactive=False)
                reasoning_text = gr.Textbox(label="Why This Matters", interactive=False, lines=2)
                issues_text = gr.Textbox(label="Technical Issues", interactive=False)
                cost_text = gr.Textbox(label="Estimated Cost", interactive=False)
                status_text = gr.Textbox(label="Current Status", interactive=False)

        with gr.Row():
            reject_btn = gr.Button("❌ Reject", variant="secondary", size="lg")
            approve_btn = gr.Button("✅ Approve", variant="primary", size="lg")

        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", size="sm")
            next_btn = gr.Button("➡️ Next", size="sm")
            save_btn = gr.Button("💾 Save & Close", variant="secondary", size="sm")

        # Summary
        summary_text = gr.Markdown("")

        def update_display(index, cands):
            """Update the display for current candidate."""
            if not cands or index < 0 or index >= len(cands):
                return [None] * 10

            candidate = cands[index]

            # Load image
            img_path = candidates_folder / candidate["filename"]
            try:
                img = Image.open(img_path)
            except Exception:
                img = None

            # Count approved/rejected
            approved = sum(1 for c in cands if c.get("approved", "").upper() == "Y")
            rejected = sum(1 for c in cands if c.get("approved", "").upper() == "N")
            pending = len(cands) - approved - rejected

            summary = f"**Summary:** ✅ {approved} approved | ❌ {rejected} rejected | ⏳ {pending} pending"

            # Highlight Save & Close only when on last photo AND all photos reviewed
            is_last = (index == len(cands) - 1)
            all_reviewed = (pending == 0)
            if is_last and all_reviewed:
                save_btn_update = gr.update(variant="primary", value="💾 Save & Close ✓")
            else:
                save_btn_update = gr.update(variant="secondary", value="💾 Save & Close")

            return [
                img,
                f"**Progress:** {index + 1} / {len(cands)}",
                candidate.get("filename", ""),
                candidate.get("contextual_value", "").upper(),
                candidate.get("contextual_reasoning", ""),
                candidate.get("improvement_reasons", "").replace(",", ", "),
                candidate.get("estimated_cost", ""),
                candidate.get("approved", "pending").upper() or "PENDING",
                summary,
                save_btn_update
            ]

        def approve_current(index, cands):
            """Mark current candidate as approved."""
            if cands and 0 <= index < len(cands):
                cands[index]["approved"] = "Y"
                # Auto-advance
                new_index = min(index + 1, len(cands) - 1)
                return [new_index, cands] + list(update_display(new_index, cands))
            return [index, cands] + list(update_display(index, cands))

        def reject_current(index, cands):
            """Mark current candidate as rejected."""
            if cands and 0 <= index < len(cands):
                cands[index]["approved"] = "N"
                # Auto-advance
                new_index = min(index + 1, len(cands) - 1)
                return [new_index, cands] + list(update_display(new_index, cands))
            return [index, cands] + list(update_display(index, cands))

        def go_prev(index, cands):
            """Go to previous candidate."""
            new_index = max(0, index - 1)
            return [new_index, cands] + list(update_display(new_index, cands))

        def go_next(index, cands):
            """Go to next candidate."""
            new_index = min(len(cands) - 1, index + 1) if cands else 0
            return [new_index, cands] + list(update_display(new_index, cands))

        def save_and_close(cands):
            """Save candidates and signal close."""
            save_candidates_to_csv(cands, candidates_folder)
            approved = sum(1 for c in cands if c.get("approved", "").upper() == "Y")

            # Create flag file to signal the launcher to close the server
            flag_file = candidates_folder / ".review_complete"
            flag_file.touch()

            return f"## Saved! ✅ {approved} photos approved for improvement.\n\n**You can close this browser tab now.** Processing will continue in the terminal."

        # Wire up events
        outputs = [
            current_index, candidates_state,
            photo_display, progress_text, filename_text, contextual_value_text,
            reasoning_text, issues_text, cost_text, status_text, summary_text,
            save_btn
        ]

        approve_btn.click(
            approve_current,
            inputs=[current_index, candidates_state],
            outputs=outputs
        )

        reject_btn.click(
            reject_current,
            inputs=[current_index, candidates_state],
            outputs=outputs
        )

        prev_btn.click(
            go_prev,
            inputs=[current_index, candidates_state],
            outputs=outputs
        )

        next_btn.click(
            go_next,
            inputs=[current_index, candidates_state],
            outputs=outputs
        )

        save_btn.click(
            save_and_close,
            inputs=[candidates_state],
            outputs=[summary_text]
        )

        # Initialize display
        demo.load(
            lambda: update_display(0, candidates),
            outputs=[
                photo_display, progress_text, filename_text, contextual_value_text,
                reasoning_text, issues_text, cost_text, status_text, summary_text,
                save_btn
            ]
        )

    return demo


def launch_review_ui(candidates_folder: Path, share: bool = False):
    """
    Launch the Gradio review UI.

    Args:
        candidates_folder: Path to ImprovementCandidates folder.
        share: Whether to create a public share link.
    """
    import time

    if isinstance(candidates_folder, str):
        candidates_folder = Path(candidates_folder)

    demo = create_review_interface(candidates_folder)

    # Use a flag file to signal when to close
    flag_file = candidates_folder / ".review_complete"
    if flag_file.exists():
        flag_file.unlink()

    # Launch without blocking the thread
    demo.launch(share=share, inbrowser=True, quiet=True, prevent_thread_lock=True)

    # Poll for completion signal
    print("   Waiting for review to complete (click 'Save & Close' when done)...")
    try:
        while True:
            time.sleep(0.5)
            if flag_file.exists():
                flag_file.unlink()
                time.sleep(0.3)  # Brief pause for final UI update
                break
    except KeyboardInterrupt:
        print("\n   Review cancelled by user.")

    # Shut down the server
    try:
        demo.close()
    except Exception:
        pass  # Ignore errors during shutdown


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python review_ui.py <candidates_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    launch_review_ui(folder)
