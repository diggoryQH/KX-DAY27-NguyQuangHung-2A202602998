"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review Agent", layout="wide", page_icon="🤖")
st.title("🤖 HITL PR Review Agent")
st.caption("Human-in-the-Loop PR review driven by LangGraph, SQLite audit trail, and confidence-based routing.")


# ─── Helper: Fetch recent sessions for Sidebar ─────────────────────────────
async def _fetch_recent_sessions():
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 15
            """
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


# ─── Sidebar — recent sessions & bonus tools ──────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")
    try:
        sessions = asyncio.run(_fetch_recent_sessions())
        if sessions:
            for s in sessions:
                risk = s.get("worst_risk", "low")
                risk_icon = "🟢" if risk == "low" else "🟡" if risk == "med" else "🔴"
                short_thread = s["thread_id"][:8]
                pr_num = s["pr_url"].rstrip("/").split("/")[-1]
                btn_label = f"{risk_icon} PR #{pr_num} ({short_thread}..)"
                
                with st.expander(btn_label):
                    st.write(f"**PR:** {s['pr_url']}")
                    st.write(f"**Events:** {s['events']} | **Risk:** {risk}")
                    st.caption(f"Last: {s['last_event']}")
                    if st.button("Load thread", key=f"btn_{s['thread_id']}"):
                        st.session_state.thread_id = s["thread_id"]
                        st.session_state.pr_url = s["pr_url"]
                        st.session_state.interrupt_payload = None
                        st.session_state.final = None
                        st.rerun()
        else:
            st.caption("No audit sessions yet.")
    except Exception as e:
        st.caption(f"Could not load sessions: {e}")

    st.markdown("---")
    st.header("✨ Bonus Tools")
    with st.expander("📊 Confidence Calibration"):
        try:
            async def _quick_stats():
                async with db_conn() as conn:
                    async with conn.execute(
                        "SELECT COUNT(DISTINCT thread_id), AVG(confidence) FROM audit_events WHERE action IN ('analyze', 'synthesize')"
                    ) as cur:
                        row = await cur.fetchone()
                        return row[0], row[1]
            tot_sess, avg_c = asyncio.run(_quick_stats())
            st.metric("Total Sessions", tot_sess or 0)
            st.metric("Avg Confidence", f"{(avg_c or 0):.1%}")
            st.caption("CLI tool: `python -m audit.calibrate`")
        except Exception:
            pass

    with st.expander("⏳ Time-Travel Inspector"):
        st.write("Inspect checkpointer timeline or resume from previous checkpoint:")
        st.code("python bonus/time_travel.py --thread <id>")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("🚀 Run Review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"🟡 Approval Requested — Confidence {conf:.0%}")
    st.info(f"**Reasoning:** {payload.get('confidence_reasoning', '')}")
    st.markdown(f"### Summary\n{payload.get('summary', '')}")

    comments = payload.get("comments", [])
    if comments:
        st.markdown("### Proposed Review Comments")
        for c in comments:
            st.markdown(f"- **[{c['severity'].upper()}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    if payload.get("diff_preview"):
        with st.expander("📄 Diff Preview"):
            st.code(payload["diff_preview"], language="diff")

    st.markdown("---")
    feedback = st.text_input("Feedback or reviewer note (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)

    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit & Request Changes", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🔴 Strong Escalation — Confidence {conf:.0%}")
    st.warning(f"**Reasoning:** {payload.get('confidence_reasoning', '')}")

    risks = payload.get("risk_factors", [])
    if risks:
        st.error("**Identified Risk Factors:**\n- " + "\n- ".join(risks))

    st.markdown(f"### Summary\n{payload.get('summary', '')}")

    questions = payload.get("questions", [])
    st.markdown("### ❓ Clarifying Questions for Reviewer")
    st.markdown("The agent detected high-risk code or architectural ambiguities. Please provide answers to refine the review:")

    with st.form("escalation_form"):
        answers: dict[str, str] = {}
        for idx, q in enumerate(questions):
            ans = st.text_input(f"**Q{idx + 1}:** {q}", key=f"q_{idx}")
            answers[q] = ans
        
        submitted_answers = st.form_submit_button("📤 Submit Answers & Synthesize Review", type="primary")
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("🤖 Agent fetching PR diff and analyzing with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("🔄 Resuming agent execution with human feedback..."):
            result = asyncio.run(
                run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                )
            )
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    
    st.markdown("---")
    st.header("🎯 Review Complete")

    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"🎉 **Status:** `{action}` — Review comment successfully posted to GitHub!")
        st.link_button("🌐 View Comment on GitHub PR", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("⚠️ **Status:** `rejected` — Reviewer rejected the proposed review. No comment was posted.")
    else:
        st.info(f"ℹ️ **Status:** `{action}`")

    if "analysis" in final:
        a = final["analysis"]
        st.metric(label="Final Confidence", value=f"{a.confidence:.0%}")
        with st.expander("📝 View Final PR Analysis"):
            st.write(f"**Summary:** {a.summary}")
            for c in a.comments:
                st.write(f"- [{c.severity}] `{c.file}:{c.line or '?'}` — {c.body}")

    st.markdown(
        f"**Thread ID:** `{st.session_state.thread_id}`  \n"
        f"🔍 Replay CLI: `python -m audit.replay --thread {st.session_state.thread_id}`"
    )
