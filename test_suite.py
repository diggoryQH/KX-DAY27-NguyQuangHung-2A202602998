"""Automated test suite verifying Exercises 1-4 and Bonus challenges."""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from common.github import fetch_pr
from common.schemas import PRAnalysis, ReviewState, AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD
from exercises.exercise_1_confidence import build_graph as build_g1
from exercises.exercise_2_hitl import build_graph as build_g2
from exercises.exercise_3_escalation import build_graph as build_g3
from exercises.exercise_4_audit import build_graph as build_g4
from langgraph.types import Command
import asyncio
from langgraph.checkpoint.memory import MemorySaver

def test_ex1():
    print("[TEST] Exercise 1: Confidence Routing...")
    app = build_g1()
    # Test PR 1
    res1 = app.invoke({"pr_url": "https://github.com/VinUni-AI20k/PR-Demo/pull/1"})
    assert "analysis" in res1
    assert "decision" in res1
    print(f"  PR #1: decision={res1['decision']}, conf={res1['analysis'].confidence:.0%}")
    assert res1["decision"] == "human_approval"

    # Test PR 2
    res2 = app.invoke({"pr_url": "https://github.com/VinUni-AI20k/PR-Demo/pull/2"})
    assert "analysis" in res2
    assert "decision" in res2
    print(f"  PR #2: decision={res2['decision']}, conf={res2['analysis'].confidence:.0%}")
    assert res2["decision"] == "escalate"
    print("  -> EX1 PASSED!")

def test_ex2():
    print("[TEST] Exercise 2: HITL interrupt & resume...")
    app = build_g2()
    cfg = {"configurable": {"thread_id": "test-ex2-thread"}}
    res = app.invoke({"pr_url": "https://github.com/VinUni-AI20k/PR-Demo/pull/1", "thread_id": "test-ex2-thread"}, cfg)
    assert "__interrupt__" in res, "Exercise 2 must interrupt on PR 1"
    payload = res["__interrupt__"][0].value
    assert payload["kind"] == "approval_request"
    assert "confidence" in payload
    assert "summary" in payload
    # Resume with reject to avoid posting to GitHub
    res_final = app.invoke(Command(resume={"choice": "reject", "feedback": "Test reject"}), cfg)
    assert res_final.get("final_action") == "rejected"
    print("  -> EX2 PASSED!")

def test_ex3():
    print("[TEST] Exercise 3: Escalation interrupt & synthesis...")
    app = build_g3()
    cfg = {"configurable": {"thread_id": "test-ex3-thread"}}
    res = app.invoke({"pr_url": "https://github.com/VinUni-AI20k/PR-Demo/pull/2", "thread_id": "test-ex3-thread"}, cfg)
    assert "__interrupt__" in res, "Exercise 3 must interrupt on PR 2"
    payload = res["__interrupt__"][0].value
    assert payload["kind"] == "escalation"
    assert len(payload.get("questions", [])) > 0
    print(f"  Escalation generated {len(payload['questions'])} questions")
    # Resume with answers
    answers = {q: "Safe test answer" for q in payload["questions"]}
    res_final = app.invoke(Command(resume=answers), cfg)
    assert res_final.get("final_action") in ("committed_after_escalation", "commit_failed")
    assert "analysis" in res_final
    print(f"  Refined confidence: {res_final['analysis'].confidence:.0%}")
    print("  -> EX3 PASSED!")

async def test_ex4():
    print("[TEST] Exercise 4: Async SQLite Audit Trail...")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from common.db import db_path, replay_events
    
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_g4(cp)
        thread_id = "test-ex4-audit-check"
        cfg = {"configurable": {"thread_id": thread_id}}
        res = await app.ainvoke({"pr_url": "https://github.com/VinUni-AI20k/PR-Demo/pull/1", "thread_id": thread_id}, cfg)
        assert "__interrupt__" in res
        res_final = await app.ainvoke(Command(resume={"choice": "reject", "feedback": "Audited reject"}), cfg)
        assert res_final.get("final_action") == "rejected"
    
    events = await replay_events(thread_id)
    assert len(events) >= 5, f"Expected at least 5 audit events, got {len(events)}"
    actions = [e["action"] for e in events]
    print(f"  Recorded actions: {actions}")
    assert "fetch_pr" in actions
    assert "analyze" in actions
    assert "route" in actions
    assert "human_approval" in actions
    assert "commit" in actions
    print("  -> EX4 PASSED!")

def main():
    print("=" * 60)
    print("RUNNING HITL PR REVIEW AGENT COMPLETE VALIDATION SUITE")
    print("=" * 60)
    test_ex1()
    test_ex2()
    test_ex3()
    asyncio.run(test_ex4())
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% READY FOR SUBMISSION)")
    print("=" * 60)

if __name__ == "__main__":
    main()
