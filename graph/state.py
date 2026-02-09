"""
Omvyx Voice — Graph State Definition

Unified Entity Resolution state.  The client_profile dict is the single
source of truth for everything we know about the caller.  Fields are
hydrated from CRM lookups and progressively filled via slot-filling.

LangGraph persists this via its Checkpointer between stateless HTTP requests,
keyed by Retell's call_id (used as the LangGraph thread_id).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Required fields — the deterministic checklist the router iterates over.
# Order matters: identity first, then contact, then intent.
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: list[str] = ["dni", "name", "intent"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ClientProfile(BaseModel):
    """
    Unified client entity object.  Replaces the old loose UserProfile +
    user_found + missing_fields trio with a single coherent structure.

    identity_key:       The primary identifier used to locate this client
                        in the CRM (DNI, email, or phone).
    is_verified:        True once CRM lookup confirmed the identity.
    is_new_customer:    True if the client was NOT found in CRM.
    profile_data:       All known profile fields (name, email, phone,
                        address, loyalty_level, etc.).
    interaction_history: Past tickets / orders pulled from CRM.
    """
    identity_key: str | None = None
    is_verified: bool = False
    is_new_customer: bool = True
    profile_data: dict[str, Any] = Field(default_factory=dict)
    interaction_history: list[dict[str, Any]] = Field(default_factory=list)


class BookingRequest(BaseModel):
    """Temporary holder for an in-progress appointment booking."""
    requested_date: str | None = None
    confirmed_slot: str | None = None
    status: Literal["idle", "checking", "offered", "confirmed", "failed"] = "idle"


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class OmvyxState(MessagesState):
    """
    Extends LangGraph's built-in MessagesState (which gives us `messages`
    with the standard append-reducer) and adds domain-specific fields.

    Key design decisions:
    - client_profile: single entity object — replaces user_found, user_profile,
      and the old missing_fields list.
    - missing_required_fields: dynamically computed list of fields still
      needed before the call can proceed.  The deterministic checklist
      router skips any field already present in client_profile.profile_data.
    - system_initialized: ensures the system prompt is injected exactly
      once inside the graph (not prepended on every webhook call).
    """

    # Call metadata
    call_id: str = ""

    # Unified client entity
    client_profile: ClientProfile = Field(default_factory=ClientProfile)

    # Slot-filling control — computed dynamically from client_profile
    missing_required_fields: list[str] = Field(
        default_factory=lambda: list(REQUIRED_FIELDS)
    )
    current_slot: str = ""
    interrupted_by_faq: bool = False

    # Booking
    booking: BookingRequest = Field(default_factory=BookingRequest)

    # Routing hint set by the router node
    intent: Literal[
        "greeting", "faq", "collect_data", "booking", "end_call", "unknown"
    ] = "unknown"

    # Whether the system prompt has been injected into messages
    system_initialized: bool = False
