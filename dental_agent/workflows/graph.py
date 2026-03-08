from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from dental_agent.models.state import AppointmentState
from dental_agent.agents.supervisor import supervisor_node
from dental_agent.agents.info_agent import info_agent_node, info_tool_node
from dental_agent.agents.booking_agent import booking_agent_node, booking_tool_node
from dental_agent.agents.cancellation_agent import cancellation_agent_node, cancellation_tool_node
from dental_agent.agents.rescheduling_agent import rescheduling_agent_node, rescheduling_tool_node


def route_from_supervisor(state: AppointmentState) -> str:
    """Read next_agent from state and return the corresponding node name."""
    target = state.get("next_agent", "info_agent")
    valid = {"info_agent", "booking_agent", "cancellation_agent", "rescheduling_agent", "end"}
    return target if target in valid else "info_agent"


def _should_continue(state: AppointmentState) -> str:
    """
    If the last AI message has tool_calls, route to tool execution.
    Otherwise the agent has finished — go directly to END.
    (Avoids a redundant supervisor LLM call after every agent response.)
    """
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "end"


def build_graph():
    graph = StateGraph(AppointmentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("info_agent", info_agent_node)
    graph.add_node("info_tools", info_tool_node)
    graph.add_node("booking_agent", booking_agent_node)
    graph.add_node("booking_tools", booking_tool_node)
    graph.add_node("cancellation_agent", cancellation_agent_node)
    graph.add_node("cancellation_tools", cancellation_tool_node)
    graph.add_node("rescheduling_agent", rescheduling_agent_node)
    graph.add_node("rescheduling_tools", rescheduling_tool_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes to sub-agents
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "info_agent": "info_agent",
            "booking_agent": "booking_agent",
            "cancellation_agent": "cancellation_agent",
            "rescheduling_agent": "rescheduling_agent",
            "end": END,
        },
    )

    # Info agent loop: agent → tools → agent → END
    graph.add_conditional_edges(
        "info_agent",
        _should_continue,
        {"tools": "info_tools", "end": END},
    )
    graph.add_edge("info_tools", "info_agent")

    # Booking agent loop
    graph.add_conditional_edges(
        "booking_agent",
        _should_continue,
        {"tools": "booking_tools", "end": END},
    )
    graph.add_edge("booking_tools", "booking_agent")

    # Cancellation agent loop
    graph.add_conditional_edges(
        "cancellation_agent",
        _should_continue,
        {"tools": "cancellation_tools", "end": END},
    )
    graph.add_edge("cancellation_tools", "cancellation_agent")

    # Rescheduling agent loop
    graph.add_conditional_edges(
        "rescheduling_agent",
        _should_continue,
        {"tools": "rescheduling_tools", "end": END},
    )
    graph.add_edge("rescheduling_tools", "rescheduling_agent")

    return graph.compile()


dental_graph = build_graph()
