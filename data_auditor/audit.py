"""Step 3: scan -> plan -> validate -> REVIEW (human approval) -> approved_plan.json

The graph PAUSES at the review node using LangGraph's interrupt(), saves its state,
and resumes when you answer. Safe fixes are pre-approved; everything else needs a yes.

Usage:  python audit.py yourfile.csv
Uses the same provider settings as graph.py (GROQ_API_KEY / GOOGLE_API_KEY + LLM_PROVIDER / LLM_MODEL).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import checks
import planner
from graph import get_llm  # reuse provider selection from step 2

MAX_ATTEMPTS = 3


class State(TypedDict, total=False):
    # Only plain JSON-friendly data lives in state, so it can be saved at the pause.
    path: str
    issues: list
    plan: dict
    errors: list
    attempts: int
    approved: list
    rejected: list


def build_graph(llm):
    structured = llm.with_structured_output(planner.FixPlan)
    cache: dict = {}  # the DataFrame stays here, outside the saved state

    def scan(state: State):
        df = checks.load(state["path"])
        cache["df"] = df
        issues = [asdict(i) for i in checks.scan(df)]
        print(f"Scan found {len(issues)} issues in {len(df):,} rows.")
        return {"issues": issues, "attempts": 0, "errors": []}

    def plan(state: State):
        issues_json = json.dumps(state["issues"], indent=1, default=str)
        msg = f"DETECTED ISSUES:\n{issues_json}\n\nTEXT COLUMN VALUES:\n{planner.column_context(cache['df'])}"
        if state["errors"]:
            msg += "\n\nYour previous plan was rejected for these reasons. Fix them:\n- " + "\n- ".join(state["errors"])
        n = state["attempts"] + 1
        print(f"Planning fixes (attempt {n})...")
        result = structured.invoke([SystemMessage(content=planner.SYSTEM_PROMPT), HumanMessage(content=msg)])
        return {"plan": result.model_dump(), "attempts": n}

    def validate(state: State):
        plan_obj = planner.FixPlan.model_validate(state["plan"])
        errors = planner.validate_plan(plan_obj, cache["df"], {i["id"] for i in state["issues"]})
        if errors:
            print(f"  Plan rejected, {len(errors)} problem(s):")
            for e in errors:
                print(f"   - {e}")
        return {"errors": errors, "plan": planner.enforce_risk(plan_obj).model_dump()}

    def route(state: State):
        return "plan" if state["errors"] and state["attempts"] < MAX_ATTEMPTS else "review"

    def review(state: State):
        fixes = state["plan"]["fixes"]
        by_id = {i["id"]: i for i in state["issues"]}

        def describe(n, f):
            return {
                "n": n,
                "action": f["action"],
                "column": f["column"],
                "rationale": f["rationale"],
                "problem": " | ".join(by_id[i]["detail"] for i in f["issue_ids"] if i in by_id),
                "replacements": [f"{r['from_value']!r} -> {r['to_value']!r}" for r in (f["replacements"] or [])],
            }

        safe = [describe(n, f) for n, f in enumerate(fixes, 1) if f["risk"] == "safe"]
        pending = [describe(n, f) for n, f in enumerate(fixes, 1) if f["risk"] != "safe"]

        # Graph pauses HERE until someone resumes it with decisions. Nothing below runs until then.
        # The answer is wrapped in {"decisions": ...} because LangGraph treats an EMPTY resume value
        # (like {}) as "no answer yet" and would stay paused.
        answer = interrupt({"summary": state["plan"]["summary"], "safe": safe, "pending": pending}) if pending else {"decisions": {}}
        decisions = answer["decisions"]

        approved, rejected = [], []
        for n, f in enumerate(fixes, 1):
            if f["risk"] == "safe" or decisions.get(str(n)) is True:
                approved.append(f)
            else:
                rejected.append(f)
        return {"approved": approved, "rejected": rejected}

    g = StateGraph(State)
    g.add_node("scan", scan)
    g.add_node("plan", plan)
    g.add_node("validate", validate)
    g.add_node("review", review)
    g.add_edge(START, "scan")
    g.add_edge("scan", "plan")
    g.add_edge("plan", "validate")
    g.add_conditional_edges("validate", route, {"plan": "plan", "review": "review"})
    g.add_edge("review", END)
    return g.compile(checkpointer=MemorySaver())


def ask(payload: dict) -> dict:
    """Terminal version of the approval screen. A web UI (step 5) would replace just this function."""
    print("\n" + payload["summary"])
    print(f"\n--- {len(payload['safe'])} safe fixes (pre-approved) ---")
    for p in payload["safe"]:
        print(f"  {p['n']:>2}. {p['action']} -> {p['column'] or 'whole table'}")
    pending = payload["pending"]
    print(f"\n--- {len(pending)} fixes need YOUR decision ---")
    print("y = approve   n = reject   a = approve all remaining   q = reject all remaining\n")

    decisions, mode = {}, None
    for p in pending:
        print(f"Fix {p['n']}: {p['action']} -> {p['column'] or 'whole table'}")
        print(f"   Problem: {p['problem']}")
        print(f"   Why:     {p['rationale']}")
        if p["replacements"]:
            print("   Changes: " + "; ".join(p["replacements"]))
        if mode is None:
            ans = ""
            while ans not in ("y", "n", "a", "q"):
                ans = input("   Approve? [y/n/a/q]: ").strip().lower()
            if ans in ("a", "q"):
                mode = ans == "a"
                decisions[str(p["n"])] = mode
            else:
                decisions[str(p["n"])] = ans == "y"
        else:
            decisions[str(p["n"])] = mode
            print(f"   -> {'approved' if mode else 'rejected'} automatically")
        print()
    return decisions


def run(path: str, llm=None, decide=ask) -> dict:
    app = build_graph(llm or get_llm())
    config = {"configurable": {"thread_id": "audit-1"}}
    app.invoke({"path": path}, config)
    snapshot = app.get_state(config)
    for _ in range(5):  # guard: a stuck resume must fail loudly, never hang
        if not snapshot.next:
            return snapshot.values
        payload = snapshot.tasks[0].interrupts[0].value
        app.invoke(Command(resume={"decisions": decide(payload)}), config)
        snapshot = app.get_state(config)
    raise RuntimeError("Graph is still paused after 5 resume attempts")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "messy_orders.csv"
    final = run(path)
    approved, rejected = final["approved"], final["rejected"]
    with open("approved_plan.json", "w", encoding="utf-8") as fh:
        json.dump({"source": path, "approved": approved, "rejected": rejected}, fh, indent=2)
    print(f"Approved {len(approved)} fixes, rejected {len(rejected)}. Saved approved_plan.json")
    if final.get("errors"):
        print("WARNING: the plan still had unresolved problems:", final["errors"])