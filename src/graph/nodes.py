from agents.planner import Planner
from agents.retriever import Retriever
from agents.grader import Grader
from agents.answerer import Answerer
from agents.verifier import Verifier
from agents.deep_agent import DeepAgent
from graph.state import State
from llm.client import get_llm

_llm = get_llm()
_retriever = Retriever()

_planner = Planner(llm=_llm)
_grader = Grader(llm=_llm)
_answerer = Answerer(llm=_llm)
_verifier = Verifier(llm=_llm)
_deep_agent = DeepAgent(llm=_llm)


def planner_node(state: State) -> dict:
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    # Increment retry_count on every pass after the first
    if state.get("rewritten_query"):
        retry_count += 1
    rewritten = _planner.plan(query=query, retry_count=retry_count)
    return {"rewritten_query": rewritten, "retry_count": retry_count}


def retriever_node(state: State) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = _retriever.retrieve(query)
    return {"retrieved_docs": docs}


def grader_node(state: State) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    filtered = _grader.grade(query=query, docs=docs)
    return {"filtered_docs": filtered}


def answerer_node(state: State) -> dict:
    query = state["query"]
    docs = state.get("filtered_docs") or state.get("retrieved_docs", [])
    answer = _answerer.answer(query=query, docs=docs)
    return {"answer": answer}


def verifier_node(state: State) -> dict:
    query = state["query"]
    docs = state.get("filtered_docs") or state.get("retrieved_docs", [])
    answer = state.get("answer", "")
    verified = _verifier.verify(query=query, docs=docs, answer=answer)
    return {"verified": verified}


def deep_agent_node(state: State) -> dict:
    """Escalation node: delegates to the DeepAgent for sophisticated multi-step retrieval."""
    query = state["query"]
    answer = _deep_agent.run(query=query)
    return {"answer": answer, "deep_agent_used": True}
