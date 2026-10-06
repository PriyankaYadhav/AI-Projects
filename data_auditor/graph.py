"""Step 2: LangGraph workflow.   scan -> plan -> validate --(errors)--> plan (retry)

Usage:  python graph.py yourfile.csv
Needs:  GROQ_API_KEY (free, default). Or GOOGLE_API_KEY with LLM_PROVIDER=gemini,
        or ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import checks
import planner

MAX_ATTEMPTS = 3


class State(TypedDict):
    path: str
    df: Any
    issues: list
    plan: Optional[planner.FixPlan]
    errors: list
    attempts: int


def build_graph(llm):
    structured = llm.with_structured_output(planner.FixPlan)

    def scan(state: State):
        df = checks.load(state["path"])
        issues = checks.scan(df)
        print(f"Scan found {len(issues)} issues in {len(df):,} rows.")
        return {"df": df, "issues": issues, "attempts": 0, "errors": []}

    def plan(state: State):
        issues_json = json.dumps([asdict(i) for i in state["issues"]], indent=1, default=str)
        msg = f"DETECTED ISSUES:\n{issues_json}\n\nTEXT COLUMN VALUES:\n{planner.column_context(state['df'])}"
        if state["errors"]:
            msg += "\n\nYour previous plan was rejected for these reasons. Fix them:\n- " + "\n- ".join(state["errors"])
        n = state["attempts"] + 1
        print(f"Planning fixes (attempt {n})...")
        result = structured.invoke([SystemMessage(content=planner.SYSTEM_PROMPT), HumanMessage(content=msg)])
        return {"plan": result, "attempts": n}

    def validate(state: State):
        ids = {i.id for i in state["issues"]}
        errors = planner.validate_plan(state["plan"], state["df"], ids)
        if errors:
            print(f"  Plan rejected, {len(errors)} problem(s):")
            for e in errors:
                print(f"   - {e}")
        return {"errors": errors, "plan": planner.enforce_risk(state["plan"])}

    def route(state: State):
        return "plan" if state["errors"] and state["attempts"] < MAX_ATTEMPTS else END

    g = StateGraph(State)
    g.add_node("scan", scan)
    g.add_node("plan", plan)
    g.add_node("validate", validate)
    g.add_edge(START, "scan")
    g.add_edge("scan", "plan")
    g.add_edge("plan", "validate")
    g.add_conditional_edges("validate", route, {"plan": "plan", END: END})
    return g.compile()


def show(state: dict) -> str:
    plan, issues = state["plan"], {i.id: i for i in state["issues"]}
    out = ["", plan.summary, ""]
    for n, f in enumerate(plan.fixes, 1):
        flag = "SAFE  " if f.risk == "safe" else "REVIEW"
        where = f.column or "whole table"
        out.append(f"{n:>2}. [{flag}] {f.action} -> {where}   (fixes {', '.join(f.issue_ids)})")
        out.append(f"      {f.rationale}")
        if f.replacements:
            out.append("      " + "; ".join(f"{r.from_value!r} -> {r.to_value!r}" for r in f.replacements))
    if state["errors"]:
        out.append(f"\nWARNING: plan still has unresolved problems after {state['attempts']} attempts.")
    return "\n".join(out)


def get_llm():
    """Free by default (Groq, no credit card). LLM_PROVIDER = groq | gemini | anthropic.
    LLM_MODEL overrides the model name for any provider."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL")
    if provider == "groq":
        from langchain_groq import ChatGroq  # reads GROQ_API_KEY
        return ChatGroq(model=model or "llama-3.3-70b-versatile", temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-sonnet-4-6", temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI  # reads GOOGLE_API_KEY
    return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash-lite", temperature=0)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "messy_orders.csv"
    llm = get_llm()
    final = build_graph(llm).invoke({"path": path})
    print(show(final))
    with open("fix_plan.json", "w", encoding="utf-8") as fh:
        fh.write(final["plan"].model_dump_json(indent=2))
    print("\nSaved fix_plan.json")