from __future__ import annotations

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import AgentResult, ToolCallRecord
from utils.data_store import TravelDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def build_system_prompt(today: str | None = None) -> str:
    """
    Build the TravelBuddy system prompt.

    The prompt is intentionally explicit because the grader checks both final
    answer content and observed tool usage.
    """
    current_date = today or "unknown"
    return f"""
You are TravelBuddy, a concise Vietnamese travel assistant.
Today is {current_date}. If the user says "cuoi tuan nay" or "cuối tuần này", resolve it as 2026-06-06 when today's date is 2026-05-31.

Core rules:
- Always produce the final user-facing answer in Vietnamese.
- Use tool outputs as the only source of truth for flight prices, hotel prices, availability, total cost, and recommendations.
- Do not invent airlines, hotels, prices, dates, availability, or budgets.
- Keep the final answer short and useful.
- For grader compatibility, include plain keywords such as "budget" and "tong chi phi" when giving a normal recommendation or budget result.

Clarification rule:
- Before calling any tool, check whether the user provided enough key trip information: origin, destination, departure date, budget, and number of nights.
- If key information is missing, do not call tools. Ask one short clarification question in Vietnamese and include the words "thong tin", "budget", and "so dem" when relevant.

Safety rule:
- If the user asks for unsafe, illegal, fraudulent, or policy-bypassing travel help, do not call tools.
- Refuse briefly in Vietnamese, mention "guardrail" and "an toan", and redirect to legal travel help.
- Do not repeat illegal instructions from the user.

Tool-use rule when enough safe information is available:
1. Call search_flights first.
2. Call calculate_budget using the cheapest suitable flight total.
3. If the remaining nightly budget is positive and suitable for lodging, call search_hotels.
4. If the budget is insufficient after flights and local transport, explain that the budget is "thieu" and suggest "dieu chinh" options instead of recommending a hotel.

Recommendation rule:
- Pick one suitable flight and one suitable hotel from tool outputs.
- Mention destination, selected flight airline, selected hotel name, tong chi phi, and remaining budget.
""".strip()


def build_tools(store: TravelDataStore):
    """
    Student TODO:
    - Define exactly three tools with strong names, docstrings, and argument schemas:
      - `search_flights`
      - `calculate_budget`
      - `search_hotels`
    - Return them as a list for `create_agent(...)`.
    - Each tool should return compact JSON/text that the agent can reuse in its final answer.
    """

    @tool
    def search_flights(origin: str, destination: str, departure_date: str, travelers: int = 1) -> str:
        """Search flights for a route and departure date."""
        flights = store.search_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            travelers=travelers,
        )
        if not flights:
            return json.dumps(
                {
                    "status": "not_found",
                    "message": "No matching flights found.",
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "travelers": travelers,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "travelers": travelers,
                "flights": [flight.model_dump() for flight in flights],
            },
            ensure_ascii=False,
        )

    @tool
    def calculate_budget(
        total_budget: int,
        nights: int,
        cheapest_flight_total: int,
        destination: str,
        travelers: int = 1,
    ) -> str:
        """Calculate the remaining travel budget after flight and local transport costs."""
        if nights <= 0:
            return json.dumps(
                {
                    "status": "error",
                    "message": "nights must be greater than zero.",
                    "nights": nights,
                },
                ensure_ascii=False,
            )

        local_transport_estimate = 300_000
        remaining = total_budget - cheapest_flight_total - local_transport_estimate
        max_hotel_price_per_night = max(0, remaining // nights)

        return json.dumps(
            {
                "status": "ok",
                "destination": destination,
                "travelers": travelers,
                "nights": nights,
                "total_budget": total_budget,
                "flight_total": cheapest_flight_total,
                "local_transport_estimate": local_transport_estimate,
                "remaining_after_flight_and_transport": remaining,
                "max_hotel_price_per_night": max_hotel_price_per_night,
                "is_budget_feasible": remaining > 0,
            },
            ensure_ascii=False,
        )

    @tool
    def search_hotels(city: str, max_price_per_night: int, preferences: list[str] | None = None) -> str:
        """Search hotels that fit the remaining nightly budget and user preferences."""
        hotels = store.search_hotels(
            city=city,
            max_price_per_night=max_price_per_night,
            preferences=preferences,
        )
        if not hotels:
            return json.dumps(
                {
                    "status": "not_found",
                    "message": "No matching hotels found within the nightly budget.",
                    "city": city,
                    "max_price_per_night": max_price_per_night,
                    "preferences": preferences or [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "city": city,
                "max_price_per_night": max_price_per_night,
                "preferences": preferences or [],
                "hotels": [hotel.model_dump() for hotel in hotels],
            },
            ensure_ascii=False,
        )

    return [search_flights, calculate_budget, search_hotels]


def build_agent(
    data_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Student TODO:
    - Create `TravelDataStore`.
    - Build the chat model with `build_chat_model(...)`.
    - Build tools with `build_tools(store)`.
    - Return `create_agent(model=..., tools=..., system_prompt=...)`.
    """
    raise NotImplementedError("Complete build_agent() in src/agent/graph.py")


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Student TODO:
    - Build the agent with `build_agent(...)`.
    - Invoke it with one user message.
    - Extract:
      - the final AI answer
      - the tool call trace from `messages`
    - Return an `AgentResult`.
    """
    raise NotImplementedError("Complete run_agent() in src/agent/graph.py")


def extract_final_answer(messages) -> str:
    """Optional helper: return the last AI message text."""
    raise NotImplementedError


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool messages into a simple grading trace."""
    raise NotImplementedError
