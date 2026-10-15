"""Bonus Challenge 2 — Confidence Calibration Analysis.

Analyzes the recorded `audit_events` database to compute calibration metrics:
- Average confidence across sessions
- Approval rate vs Confidence buckets
- Assessment of whether the LLM reviewer is over- or under-confident.

Usage:
    python -m audit.calibrate
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common.db import db_conn

console = Console()


async def analyze_calibration() -> None:
    async with db_conn() as conn:
        # 1. Overall confidence stats
        async with conn.execute(
            """
            SELECT COUNT(DISTINCT thread_id) AS total_sessions,
                   COUNT(*)                   AS total_events,
                   AVG(confidence)            AS avg_conf,
                   MIN(confidence)            AS min_conf,
                   MAX(confidence)            AS max_conf
              FROM audit_events
             WHERE action IN ('analyze', 'synthesize')
            """
        ) as cur:
            overall = dict(await cur.fetchone())

        # 2. Human choices count
        async with conn.execute(
            """
            SELECT decision,
                   COUNT(*) AS count,
                   AVG(confidence) AS avg_conf_at_decision
              FROM audit_events
             WHERE action = 'human_approval' AND reviewer_id IS NOT NULL
             GROUP BY decision
            """
        ) as cur:
            decisions = [dict(r) for r in await cur.fetchall()]

        # 3. Confidence vs Decision correlation
        async with conn.execute(
            """
            SELECT CASE
                     WHEN confidence >= 0.73 THEN 'High (>=73%)'
                     WHEN confidence >= 0.58 THEN 'Medium (58-72%)'
                     ELSE 'Low (<58%)'
                   END AS bucket,
                   COUNT(*) AS total_routed,
                   SUM(CASE WHEN decision = 'auto_approve' OR decision = 'approve' THEN 1 ELSE 0 END) AS approved_count,
                   SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS rejected_count
              FROM audit_events
             WHERE action IN ('route', 'human_approval')
             GROUP BY bucket
             ORDER BY AVG(confidence) DESC
            """
        ) as cur:
            buckets = [dict(r) for r in await cur.fetchall()]

    console.rule("[bold]Bonus 2 — LLM Confidence Calibration Report[/bold]")

    # Overall Summary Table
    t_summary = Table(title="Overall Review Statistics")
    t_summary.add_column("Metric", style="cyan")
    t_summary.add_column("Value", style="bold green")

    t_summary.add_row("Total Review Sessions", str(overall["total_sessions"] or 0))
    t_summary.add_row("Total Audit Events", str(overall["total_events"] or 0))
    t_summary.add_row("Average Confidence", f"{(overall['avg_conf'] or 0):.1%}")
    t_summary.add_row("Min Confidence", f"{(overall['min_conf'] or 0):.1%}")
    t_summary.add_row("Max Confidence", f"{(overall['max_conf'] or 0):.1%}")
    console.print(t_summary)

    # Human Decisions Table
    if decisions:
        t_dec = Table(title="Human Decision Breakdown")
        t_dec.add_column("Decision", style="magenta")
        t_dec.add_column("Count", justify="right")
        t_dec.add_column("Avg Confidence at Decision", justify="right")
        for d in decisions:
            t_dec.add_row(
                d["decision"].upper(),
                str(d["count"]),
                f"{(d['avg_conf_at_decision'] or 0):.1%}",
            )
        console.print(t_dec)

    # Confidence Buckets Table
    if buckets:
        t_buck = Table(title="Confidence Buckets & Approval Rates")
        t_buck.add_column("Confidence Bucket", style="yellow")
        t_buck.add_column("Total Events", justify="right")
        t_buck.add_column("Approved", justify="right", style="green")
        t_buck.add_column("Rejected", justify="right", style="red")
        for b in buckets:
            t_buck.add_row(
                b["bucket"],
                str(b["total_routed"]),
                str(b["approved_count"]),
                str(b["rejected_count"]),
            )
        console.print(t_buck)

    # Assessment logic
    avg_conf = overall["avg_conf"] or 0.0
    total_approved = sum(d["count"] for d in decisions if d["decision"] == "approve")
    total_reviewed = sum(d["count"] for d in decisions)
    approval_rate = (total_approved / total_reviewed) if total_reviewed > 0 else 1.0

    if total_reviewed == 0:
        diagnosis = (
            "Not enough human feedback recorded yet to establish statistical confidence calibration.\n"
            "Run more PR reviews via CLI or Streamlit to collect audit data."
        )
        panel_color = "yellow"
    elif avg_conf > 0.80 and approval_rate < 0.50:
        diagnosis = (
            "⚠️ OVER-CONFIDENT: The LLM model is assigning high confidence (>80%), "
            "but human reviewers are rejecting or editing the reviews frequently. "
            "Recommendation: Tighten system prompts or raise thresholds."
        )
        panel_color = "red"
    elif avg_conf < 0.50 and approval_rate > 0.80:
        diagnosis = (
            "⚠️ UNDER-CONFIDENT: The LLM assigns low confidence (<50%), "
            "yet humans approve the reviews without modification. "
            "Recommendation: Consider relaxing escalation triggers."
        )
        panel_color = "yellow"
    else:
        diagnosis = (
            "✅ WELL-CALIBRATED: The LLM confidence scores align well with human reviewer outcomes. "
            "Medium-risk PRs undergo human scrutiny as designed, and high-risk PRs properly escalate."
        )
        panel_color = "green"

    console.print(Panel(diagnosis, title="Calibration Assessment", border_style=panel_color))


def main():
    asyncio.run(analyze_calibration())


if __name__ == "__main__":
    main()
