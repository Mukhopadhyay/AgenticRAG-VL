from langgraph import graph
from langgraph.graph import StateGraph, END, START

from graph.state import State
from graph.nodes import (
    planner_node,
    retriever_node,
    grader_node,
    answerer_node,
    verifier_node,
    deep_agent_node,
)
from config import settings


def _route_after_grader(state: State) -> str:
    """Route to retry, escalate to the DeepAgent, or proceed to the Answerer."""
    has_docs = bool(state.get("filtered_docs"))
    retry_count = state.get("retry_count", 0)

    if has_docs:
        return "answer"
    if retry_count < 1:
        # First miss — give the Planner one more attempt with a retry hint.
        return "retry"
    # Second miss — escalate to the DeepAgent for richer multi-step retrieval.
    return "escalate"


def _route_after_verifier(state: State) -> str:
    """End if verified or retries are exhausted; escalate after the first failure."""
    if state.get("verified") or state.get("retry_count", 0) >= settings.MAX_RETRIES:
        return "end"
    if state.get("retry_count", 0) >= 1:
        # Already retried once — hand off to the DeepAgent.
        return "escalate"
    return "retry"


def build_workflow():
    graph = StateGraph(State)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("deep_agent", deep_agent_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges(
        "grader",
        _route_after_grader,
        {"retry": "planner", "answer": "answerer", "escalate": "deep_agent"},
    )
    graph.add_edge("answerer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"end": END, "retry": "planner", "escalate": "deep_agent"},
    )
    graph.add_edge("deep_agent", END)

    graph = graph.compile()
    mermaid_code = graph.get_graph().draw_mermaid()
    print(mermaid_code)
    return graph


workflow = build_workflow()
