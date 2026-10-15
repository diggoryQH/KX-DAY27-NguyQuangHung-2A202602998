"""Bonus Challenge 3 — Multi-Reviewer Fan-Out with LangGraph Send API.

When escalating high-risk PRs, use the `Send` API to fan-out the questions
to multiple specialized reviewers (e.g. Security Specialist + Architecture Lead),
collect all answers, and synthesize a multi-perspective review.

Usage:
    python bonus/multi_reviewer.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
"""

from __future__ import annotations

import argparse
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import PRAnalysis

load_dotenv()
console = Console()


# ─── Specialized Sub-state for Send API ────────────────────────────────────
class ExpertTask(TypedDict):
    role: str
    focus: str
    diff: str
    questions: list[str]


class ExpertFeedback(TypedDict):
    role: str
    feedback: str
    recommendation: str


class MultiReviewerState(TypedDict):
    pr_url: str
    pr_title: str
    pr_diff: str
    initial_analysis: PRAnalysis
    # Annotated with operator.add so all Send branches append to this list
    expert_feedbacks: Annotated[list[ExpertFeedback], operator.add]
    final_analysis: PRAnalysis
    final_action: str


# ─── Graph Nodes ───────────────────────────────────────────────────────────
def node_fetch_and_analyze(state: MultiReviewerState) -> dict:
    console.print("[cyan]→ fetch_and_analyze[/cyan]")
    pr = fetch_pr(state["pr_url"])
    
    llm = get_llm().with_structured_output(PRAnalysis)
    system_prompt = (
        "You are a Senior reviewer. Analyze the diff and populate escalation_questions "
        "if there are security, architectural, or reliability risks."
    )
    with console.status("[dim]Analyzing PR...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Title: {pr.title}\nDiff:\n{pr.diff}"},
        ])
    console.print(f"  [green]✓[/green] Found {len(analysis.escalation_questions)} escalation questions (conf={analysis.confidence:.0%})")
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "initial_analysis": analysis,
    }


def fan_out_to_experts(state: MultiReviewerState) -> list[Send]:
    """Conditional edge using LangGraph Send API to fan-out to 2 specialized reviewers."""
    console.print("[yellow]→ Fan-out to 2 specialized reviewers via Send API[/yellow]")
    questions = state["initial_analysis"].escalation_questions or ["General assessment"]
    diff = state["pr_diff"]

    return [
        Send("expert_review", {
            "role": "Security Specialist",
            "focus": "Vulnerabilities, authentication, hashing, injection, token storage",
            "diff": diff,
            "questions": questions,
        }),
        Send("expert_review", {
            "role": "Architecture Lead",
            "focus": "Schema migration, backward compatibility, service contracts, maintainability",
            "diff": diff,
            "questions": questions,
        }),
    ]


def node_expert_review(task: ExpertTask) -> dict:
    """Invoked independently for each Send() task."""
    role = task["role"]
    focus = task["focus"]
    console.print(f"  [magenta]⚡ [{role}] reviewing diff (Focus: {focus})[/magenta]")

    llm = get_llm()
    prompt = (
        f"You are the {role}. Your evaluation focus is: {focus}.\n"
        f"Questions under review: {task['questions']}\n"
        f"Analyze the diff and provide a brief expert assessment and recommendation."
    )
    resp = llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Diff:\n{task['diff']}"},
    ])

    return {
        "expert_feedbacks": [{
            "role": role,
            "feedback": resp.content,
            "recommendation": "Review comments provided",
        }]
    }


def node_synthesize_multi_reviews(state: MultiReviewerState) -> dict:
    """Fan-in node: combines expert feedbacks from all Send branches."""
    console.print("[cyan]→ synthesize_multi_reviews (Fan-in)[/cyan]")
    feedbacks = state["expert_feedbacks"]
    console.print(f"  [green]✓[/green] Collected {len(feedbacks)} expert evaluations:")
    for fb in feedbacks:
        console.print(Panel(fb["feedback"][:250] + "...", title=fb["role"], border_style="cyan"))

    llm = get_llm().with_structured_output(PRAnalysis)
    summary_of_experts = "\n\n".join(f"### {fb['role']}\n{fb['feedback']}" for fb in feedbacks)
    
    with console.status("[dim]Synthesizing combined PR review...[/dim]"):
        refined = llm.invoke([
            {"role": "system", "content": "Produce structured PRAnalysis combining security and architecture expert feedback."},
            {"role": "user", "content": f"Diff:\n{state['pr_diff']}\n\nExpert Reports:\n{summary_of_experts}"},
        ])

    console.print(f"  [green]✓[/green] Combined Review Confidence: {refined.confidence:.0%}")
    return {
        "final_analysis": refined,
        "final_action": "multi_expert_review_committed",
    }


def build_multi_reviewer_graph():
    g = StateGraph(MultiReviewerState)
    g.add_node("fetch_and_analyze", node_fetch_and_analyze)
    g.add_node("expert_review", node_expert_review)
    g.add_node("synthesize_multi_reviews", node_synthesize_multi_reviews)

    g.add_edge(START, "fetch_and_analyze")
    g.add_conditional_edges("fetch_and_analyze", fan_out_to_experts, ["expert_review"])
    g.add_edge("expert_review", "synthesize_multi_reviews")
    g.add_edge("synthesize_multi_reviews", END)

    return g.compile(checkpointer=MemorySaver())


def main():
    parser = argparse.ArgumentParser(description="Bonus 3: Multi-Reviewer Fan-Out with LangGraph Send API")
    parser.add_argument("--pr", default="https://github.com/VinUni-AI20k/PR-Demo/pull/2")
    args = parser.parse_args()

    console.rule("[bold]Bonus 3 — Multi-Reviewer Fan-Out (LangGraph Send API)[/bold]")
    app = build_multi_reviewer_graph()
    cfg = {"configurable": {"thread_id": "multi-reviewer-demo"}}
    result = app.invoke({"pr_url": args.pr}, cfg)

    console.rule("Final Multi-Review Result")
    console.print(f"Action: [bold green]{result.get('final_action')}[/bold green]")
    a = result["final_analysis"]
    console.print(f"Combined Confidence: {a.confidence:.0%}")
    console.print(f"Summary: {a.summary}")


if __name__ == "__main__":
    main()
