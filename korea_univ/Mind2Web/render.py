"""Render a Mind2Web step's saved HTML and screenshot it with the ground-truth and
model-predicted target elements highlighted.

Mind2Web is an offline dataset -- each step's `cleaned_html` is a static snapshot, not
a live page, so there is nothing to "interact" with. This module renders that snapshot
in a headless browser purely to produce a visual, capturable record of what the model
picked vs. what the correct answer was (green outline = ground truth, green/red outline
= model's pick depending on correctness).
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

# same viewport convention as WebArena's run.py (EnvArgs viewport)
VIEWPORT = {"width": 1500, "height": 1280}


class Mind2WebRenderer:
    """Wraps a single reused browser/page (launching Playwright per screenshot would be
    far too slow across hundreds of steps)."""

    def __init__(self, headless: bool = True):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._page = self._browser.new_page(viewport=VIEWPORT)

    def screenshot(
        self,
        cleaned_html: str,
        out_path: Path,
        gt_backend_node_id: str | None,
        pred_backend_node_id: str | None,
        pred_correct: bool,
    ) -> bool:
        """Render the step HTML, highlight the ground-truth/predicted elements, save a
        screenshot. Returns True if at least one highlighted element was actually found
        on the rendered page (a miss usually means the id doesn't exist in this step's
        HTML -- e.g. a hallucinated prediction)."""
        self._page.set_content(cleaned_html, wait_until="load")

        pred_color = "#2ea043" if pred_correct else "#e5484d"
        style_rules = []
        if gt_backend_node_id:
            style_rules.append(
                f'[backend_node_id="{gt_backend_node_id}"] {{'
                "outline: 4px solid #2ea043 !important; outline-offset: -2px !important;"
                "background-color: rgba(46, 160, 67, 0.15) !important; }"
            )
        if pred_backend_node_id and pred_backend_node_id != gt_backend_node_id:
            style_rules.append(
                f'[backend_node_id="{pred_backend_node_id}"] {{'
                f"outline: 4px solid {pred_color} !important; outline-offset: -2px !important;"
                f"box-shadow: 0 0 0 2px {pred_color} !important; }}"
            )
        if style_rules:
            self._page.add_style_tag(content="\n".join(style_rules))

        found = False
        for node_id in filter(None, dict.fromkeys([gt_backend_node_id, pred_backend_node_id])):
            locator = self._page.locator(f'[backend_node_id="{node_id}"]')
            if locator.count() > 0:
                found = True
                try:
                    locator.first.scroll_into_view_if_needed(timeout=2000)
                except Exception:
                    pass
                break

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # full_page so both the ground-truth and predicted highlights are visible
        # regardless of which one the initial scroll position landed on
        self._page.screenshot(path=str(out_path), full_page=True)
        return found

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()

    def __enter__(self) -> "Mind2WebRenderer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
