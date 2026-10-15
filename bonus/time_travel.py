"""Bonus Challenge 1 — Time-Travel with LangGraph Checkpointer.

Use `app.aget_state_history(config)` to inspect every historical checkpoint
and time-travel back to resume with a different human decision.

Usage:
    python bonus/time_travel.py --thread <thread_id>
    python bonus/time_travel.py --thread <thread_id> --resume-with reject --feedback "Code needs unit tests"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from common.db import db_path
from exercises.exercise_4_audit import build_graph

load_dotenv()
console = Console()


async def list_history(thread_id: str) -> None:
    """List every checkpoint in this thread's history."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        table = Table(title=f"Checkpoint History for Thread {thread_id}")
        table.add_column("Step", style="dim", width=6)
        table.add_column("Checkpoint ID", style="cyan")
        table.add_column("Next Node", style="magenta")
        table.add_column("Confidence", justify="right")
        table.add_column("Decision / Action", style="green")
        table.add_column("Interrupt?", justify="center")

        step = 0
        async for state in app.aget_state_history(cfg):
            step += 1
            snap_cfg = state.config.get("configurable", {})
            checkpoint_id = snap_cfg.get("checkpoint_id", "-")
            next_node = ", ".join(state.next) if state.next else "(END)"
            
            analysis = state.values.get("analysis")
            conf_str = f"{analysis.confidence:.0%}" if analysis else "-"
            
            action = state.values.get("final_action") or state.values.get("decision") or "-"
            has_interrupt = "YES" if state.tasks and any(t.interrupts for t in state.tasks) else "no"

            table.add_row(
                str(step),
                checkpoint_id,
                next_node,
                conf_str,
                str(action),
                f"[yellow]{has_interrupt}[/yellow]" if has_interrupt == "YES" else "[dim]no[/dim]",
            )

        console.print(table)


async def time_travel_resume(thread_id: str, new_choice: str, feedback: str = "") -> None:
    """Resume a paused checkpoint with an alternative reviewer decision."""
    console.rule(f"[bold]Time-Travel Resuming Thread {thread_id}[/bold]")
    console.print(f"Injecting new choice: [bold cyan]{new_choice}[/bold cyan] (Feedback: '{feedback}')")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # Find the checkpoint where the graph was paused at interrupt
        target_checkpoint = None
        async for state in app.aget_state_history(cfg):
            if state.tasks and any(t.interrupts for t in state.tasks):
                target_checkpoint = state
                break

        if not target_checkpoint:
            console.print("[yellow]No active interrupt found in latest checkpoint history. Resuming from current state...[/yellow]")
            resume_cfg = cfg
        else:
            checkpoint_id = target_checkpoint.config["configurable"]["checkpoint_id"]
            console.print(f"[green]Found interrupt checkpoint:[/green] [dim]{checkpoint_id}[/dim]")
            resume_cfg = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

        answer = {"choice": new_choice, "feedback": feedback}
        result = await app.ainvoke(Command(resume=answer), resume_cfg)
        console.rule("Time-Travel Result")
        console.print(f"Final Action: [bold green]{result.get('final_action')}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonus 1: Time-Travel Checkpoint Inspector")
    parser.add_argument("--thread", required=True, help="Thread ID to inspect")
    parser.add_argument("--resume-with", choices=["approve", "reject", "edit"], help="Alternative decision to resume with")
    parser.add_argument("--feedback", default="", help="Optional feedback note")
    args = parser.parse_args()

    if args.resume_with:
        asyncio.run(time_travel_resume(args.thread, args.resume_with, args.feedback))
    else:
        asyncio.run(list_history(args.thread))


if __name__ == "__main__":
    main()
