from langgraph.graph import StateGraph, END, START

from graph.state import State
from graph.nodes import (
    planner_node,
    retriever_node,
    grader_node,
    answerer_node,
    verifier_node,
)
from config import settings


def _route_after_grader(state: State) -> str:
    """Retry if no relevant docs were found and we still have attempts left."""
    if (
        not state.get("filtered_docs")
        and state.get("retry_count", 0) < settings.MAX_RETRIES
    ):
        return "retry"
    return "answer"


def _route_after_verifier(state: State) -> str:
    """End if the answer is verified or retries are exhausted."""
    if state.get("verified") or state.get("retry_count", 0) >= settings.MAX_RETRIES:
        return "end"
    return "retry"


def build_workflow():
    graph = StateGraph(State)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("verifier", verifier_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges(
        "grader",
        _route_after_grader,
        {"retry": "planner", "answer": "answerer"},
    )
    graph.add_edge("answerer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"end": END, "retry": "planner"},
    )

    return graph.compile()


workflow = build_workflow()
