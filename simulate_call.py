#!/usr/bin/env python3
"""
Omvyx Voice — Call Simulator (Entity Resolution Architecture)

Runs a multi-turn conversation against the LangGraph workflow directly
(no HTTP server needed).  Demonstrates:

    1. Greeting
    2. Slot-filling with deterministic checklist (DNI first → CRM sync)
    3. CRM hydration for known customers (skips already-known fields)
    4. FAQ interruption mid-slot-filling (and resumption)
    5. Appointment booking
    6. Goodbye

Two scenarios:
    - NEW CUSTOMER: DNI not in CRM → full slot-filling
    - KNOWN CUSTOMER: DNI in CRM → hydrated from DB, skips known fields

Usage:
    python simulate_call.py              # run the default scripted scenario
    python simulate_call.py --known      # run the known-customer scenario
    python simulate_call.py --interactive  # interactive REPL mode
"""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage

from graph.workflow import compile_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALL_ID_NEW = "sim-call-new-001"
CALL_ID_KNOWN = "sim-call-known-001"


def _last_ai_message(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            return msg.content
    return "(no response)"


async def send(graph, text: str, *, config: dict, call_id: str) -> str:
    """Send a user utterance and return the agent reply."""
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=text)],
            "call_id": call_id,
        },
        config=config,
    )
    return _last_ai_message(result)


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_state(vals: dict):
    """Pretty-print the final state snapshot."""
    profile = vals.get("client_profile", {})
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()
    print(f"  client_profile:")
    print(f"    identity_key       : {profile.get('identity_key')}")
    print(f"    is_verified        : {profile.get('is_verified')}")
    print(f"    is_new_customer    : {profile.get('is_new_customer')}")
    print(f"    profile_data       : {profile.get('profile_data')}")
    history = profile.get('interaction_history', [])
    print(f"    interaction_history : {len(history)} entries")
    for h in history:
        print(f"      - {h}")
    print(f"  missing_required     : {vals.get('missing_required_fields')}")
    booking = vals.get("booking", {})
    if hasattr(booking, "model_dump"):
        booking = booking.model_dump()
    print(f"  booking              : {booking}")


# ---------------------------------------------------------------------------
# Scenario 1: NEW CUSTOMER (DNI not in CRM)
# ---------------------------------------------------------------------------

SCENARIO_NEW: list[tuple[str, str]] = [
    ("1. Greeting", "Hola, buenos días"),
    ("2. Provide DNI (unknown)", "Mi DNI es 99887766C"),
    ("3. Provide name", "Me llamo Ana Torres"),
    ("4. FAQ interruption", "¿Dónde están ubicados?"),
    ("5. Request appointment", "Quiero una cita para 2026-02-09 10:00"),
    ("6. Pick alternative", "Mejor el 2026-02-09 12:00"),
    ("7. Goodbye", "Eso es todo, adiós"),
]


# ---------------------------------------------------------------------------
# Scenario 2: KNOWN CUSTOMER (DNI in CRM → full hydration)
# ---------------------------------------------------------------------------

SCENARIO_KNOWN: list[tuple[str, str]] = [
    ("1. Greeting with DNI", "Hola, mi DNI es 12345678A"),
    ("2. Request booking", "Quiero reservar una cita para 2026-02-11 10:00"),
    ("3. Goodbye", "Gracias, adiós"),
]


async def run_scenario(name: str, scenario: list, call_id: str):
    banner(f"OMVYX VOICE — {name}")
    graph = compile_graph()
    config = {"configurable": {"thread_id": call_id}}

    for label, utterance in scenario:
        print(f"\n--- {label} ---")
        print(f"  USER : {utterance}")
        reply = await send(graph, utterance, config=config, call_id=call_id)
        print(f"  AGENT: {reply}")

    # Print final state
    state = await graph.aget_state(config)
    print("\n" + "=" * 60)
    print("  FINAL STATE SNAPSHOT")
    print("=" * 60)
    print_state(state.values)
    print()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def run_interactive():
    banner("OMVYX VOICE — Interactive Mode (type 'quit' to exit)")
    graph = compile_graph()
    call_id = "sim-interactive"
    config = {"configurable": {"thread_id": call_id}}

    while True:
        try:
            text = input("\n  YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        reply = await send(graph, text, config=config, call_id=call_id)
        print(f"  OMVYX: {reply}")

    # Show final state
    state = await graph.aget_state(config)
    print("\n" + "=" * 60)
    print("  FINAL STATE")
    print("=" * 60)
    print_state(state.values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--interactive" in sys.argv:
        asyncio.run(run_interactive())
    elif "--known" in sys.argv:
        asyncio.run(run_scenario("Known Customer (CRM Hydration)", SCENARIO_KNOWN, CALL_ID_KNOWN))
    else:
        asyncio.run(run_scenario("New Customer (Full Slot-Filling)", SCENARIO_NEW, CALL_ID_NEW))
