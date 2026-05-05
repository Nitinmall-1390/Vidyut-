"""
VIDYUT Evaluation Report Generator
===========================================================================
Produces structured JSON + HTML evaluation reports for both demand
forecasting and theft detection model runs.
===========================================================================
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

log = get_logger("vidyut.evaluation_report")


class EvaluationReport:
    """
    Collects metrics and metadata from a model evaluation run and persists
    them as JSON (machine-readable) and HTML (human-readable).
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        task: str,  # "demand_forecasting" | "theft_detection" | "ring_detection"
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.task = task
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "model_name": model_name,
            "model_version": model_version,
            "task": task,
        }
        self.sections: list = []

    def add_metrics(self, metrics: Dict[str, Any], section: str = "main") -> None:
        """Add a named metrics dict to the report."""
        self.metrics[section] = metrics
        log.info("[%s] Metrics added for section '%s': %s", self.model_name, section, metrics)

    def add_metadata(self, **kwargs) -> None:
        """Attach arbitrary metadata key-value pairs."""
        self.metadata.update(kwargs)

    def add_section(self, title: str, content: str) -> None:
        """Add a free-text section (for HTML report narrative)."""
        self.sections.append({"title": title, "content": content})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "sections": self.sections,
        }

    def save_json(self, output_dir: Path) -> Path:
        """Persist report as JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.model_name}_{self.model_version}_{self.task}.json"
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.info("Evaluation report saved: %s", path)
        return path

    def save_html(self, output_dir: Path) -> Path:
        """Persist report as a self-contained HTML file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.model_name}_{self.model_version}_{self.task}.html"
        path = output_dir / filename

        rows = ""
        for section, section_metrics in self.metrics.items():
            rows += f"<tr><td colspan='2' style='background:#2d3748;color:#fff;padding:8px'><b>{section}</b></td></tr>\n"
            if isinstance(section_metrics, dict):
                for k, v in section_metrics.items():
                    v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    rows += f"<tr><td style='padding:6px 12px'>{k}</td><td style='padding:6px 12px'><b>{v_str}</b></td></tr>\n"

        extra_sections_html = ""
        for s in self.sections:
            extra_sections_html += (
                f"<h3>{s['title']}</h3><p style='color:#4a5568'>{s['content']}</p>"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Vidyut Evaluation — {self.model_name}</title>
<style>
  body {{font-family: 'Segoe UI', sans-serif; margin:40px; background:#f7fafc}}
  h1 {{color:#2d3748}} h2 {{color:#4a5568; border-bottom:2px solid #e2e8f0}}
  table {{border-collapse:collapse; width:60%; margin-top:20px}}
  td,th {{border:1px solid #e2e8f0; text-align:left}}
  tr:nth-child(even) {{background:#edf2f7}}
  .badge {{display:inline-block;padding:4px 12px;border-radius:20px;
           background:#48bb78;color:white;font-size:13px}}
</style>
</head>
<body>
<h1>🔋 Vidyut Evaluation Report</h1>
<p>Model: <b>{self.model_name}</b> &nbsp;|&nbsp;
   Version: <b>{self.model_version}</b> &nbsp;|&nbsp;
   Task: <span class="badge">{self.task}</span></p>
<p>Generated: {self.created_at}</p>
<h2>Metrics</h2>
<table>{rows}</table>
<h2>Metadata</h2>
<pre style="background:#2d3748;color:#e2e8f0;padding:16px;border-radius:6px">
{json.dumps(self.metadata, indent=2, default=str)}
</pre>
{extra_sections_html}
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        log.info("HTML report saved: %s", path)
        return path
