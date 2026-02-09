"""
Omvyx Voice — LangGraph Workflow (Entity Resolution Architecture)

Intelligence Hub architecture:

    ┌────────────────┐
    │  init_system    │  ← injects system prompt once
    └───────┬────────┘
            ▼
    ┌────────────────┐
    │ universal_extract│  ← scans EVERY input for profile data
    └───────┬────────┘
            ▼
    ┌────────────────┐
    │   crm_sync     │  ← hydrates client_profile from CRM on first ID
    └───────┬────────┘
            ▼
    ┌────────────────┐
    │ checklist_router│  ← deterministic: skips known fields, routes
    └───────┬────────┘
            │ conditional edges
    ┌───────┼──────────┬──────────────┐
    ▼       ▼          ▼              ▼
  greet  handle_faq  collect_data  manage_booking  end_call
    │       │          │              │               │
    └───────┴──────────┴──────────────┴───────────────┘
                           ▼
                   ┌────────────┐
                   │  save_crm  │  ← UPDATE or INSERT guard
                   └─────┬──────┘
                         ▼
                    ┌─────────┐
                    │ respond  │ → END
                    └─────────┘

CRITICAL — Persistence:
    The graph is compiled with a MemorySaver checkpointer.  Every
    invocation receives `config={"configurable": {"thread_id": call_id}}`
    so state is resumed from the exact point where the previous webhook
    request left off.

NOTE:
    LangGraph passes state as a plain dict to node functions.
    All nodes use dict-style access: state["messages"], etc.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.state import REQUIRED_FIELDS, BookingRequest, ClientProfile, OmvyxState
from tools.calendar import book_slot, check_availability
from tools.crm import create_user, lookup_user, update_user
from tools.faq import search_faq

logger = logging.getLogger("omvyx.workflow")


# ---------------------------------------------------------------------------
# System prompt — personality of the receptionist
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""\
Eres Omvyx, una recepcionista virtual profesional y amable.
Hablas en español de España de forma natural y cercana.

REGLAS:
- Sé concisa: las respuestas van por voz, así que máximo 2-3 frases.
- Nunca inventes datos. Si no sabes algo, dilo.
- Si estás recogiendo datos del usuario y te interrumpen con una pregunta,
  responde la pregunta y VUELVE INMEDIATAMENTE a pedir el dato que faltaba.
- Cuando ofrezcas citas alternativas, di las opciones de forma clara.
- Si ya conoces al cliente (CRM), salúdale por su nombre y menciona su
  historial si es relevante.
""")


# ---------------------------------------------------------------------------
# FIELD_PROMPTS — what to say when asking for each slot
# ---------------------------------------------------------------------------

FIELD_PROMPTS: dict[str, str] = {
    "dni": "Para poder buscarle en nuestro sistema, ¿me puede facilitar su DNI o documento de identidad?",
    "name": "Perfecto. ¿Me puede indicar su nombre completo?",
    "intent": "¿En qué puedo ayudarle hoy? Puedo agendar una cita, resolver dudas o consultar información.",
}


# ---------------------------------------------------------------------------
# State access helpers
# ---------------------------------------------------------------------------

def _get_profile(state: dict) -> ClientProfile:
    """Reconstruct ClientProfile from the state dict."""
    raw = state.get("client_profile")
    if isinstance(raw, ClientProfile):
        return raw
    if isinstance(raw, dict):
        return ClientProfile(**raw)
    return ClientProfile()


def _get_booking(state: dict) -> BookingRequest:
    """Reconstruct BookingRequest from the state dict."""
    raw = state.get("booking")
    if isinstance(raw, BookingRequest):
        return raw
    if isinstance(raw, dict):
        return BookingRequest(**raw)
    return BookingRequest()


def _last_human_text(state: dict) -> str:
    """Extract the text of the most recent HumanMessage."""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# ---------------------------------------------------------------------------
# Extraction helpers — pull structured data from free-text user utterances
# ---------------------------------------------------------------------------

def _extract_field_value(field: str, text: str) -> str | None:
    """
    Best-effort extraction of a slot value from the user's last message.
    In production you'd use an LLM function-call or NER model here.
    """
    text = text.strip()
    if not text:
        return None

    if field == "dni":
        m = re.search(r"\b(\d{7,8}\s*[A-Za-z])\b", text)
        return m.group(1).replace(" ", "").upper() if m else None

    if field == "email":
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        return m.group(0).lower() if m else None

    if field == "phone":
        # Strip date-like patterns before extracting digits
        cleaned = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
        cleaned = re.sub(r"\d{2}:\d{2}", "", cleaned)
        digits = re.sub(r"[^\d+]", "", cleaned)
        if len(digits) >= 9:
            return digits
        return None

    if field == "name":
        # Explicit patterns: "me llamo X", "soy X", "mi nombre es X"
        name_patterns = [
            r"(?:me llamo|soy|mi nombre es)\s+(.+)",
        ]
        for pat in name_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                name_part = m.group(1).strip().rstrip(".")
                words = [w for w in name_part.split() if w.isalpha()]
                if words:
                    return " ".join(words).title()

        # Fallback: accept if entire message looks like a name (≥2 words)
        # Exclude greetings, data-providing phrases, and common filler
        _SKIP_WORDS = {
            "hola", "buenos", "días", "dias", "buenas", "tardes", "noches",
            "hey", "hello", "hi", "gracias", "adiós", "adios", "bye",
            "sí", "si", "no", "vale", "ok", "bien", "perfecto", "genial",
            "mi", "es", "dni", "documento", "identidad", "correo", "email",
            "teléfono", "telefono", "número", "numero", "quiero", "una",
            "cita", "para", "el", "la", "de", "por", "favor", "mejor",
            "eso", "todo", "nada", "más", "mas",
        }
        words = [w for w in text.split() if w.isalpha()]
        # If any word is a "data" keyword, this isn't a name
        if any(w.lower() in _SKIP_WORDS for w in words):
            return None
        if 2 <= len(words) <= 4:
            return " ".join(words).title()
        if len(words) == 1 and len(words[0]) > 1:
            return words[0].title()
        return None

    return None


def _detect_intent(text: str) -> str:
    """Lightweight keyword-based intent classifier."""
    t = text.lower()

    greetings = ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "hello"]
    if any(g in t for g in greetings) and len(t.split()) <= 4:
        return "greeting"

    faq_signals = [
        "dónde", "donde", "ubicación", "dirección", "horario", "hora",
        "precio", "costo", "cancelar", "seguro", "parking", "location",
        "where", "address", "hours", "price", "cancel", "insurance",
    ]
    if any(s in t for s in faq_signals):
        return "faq"

    booking_signals = ["cita", "reservar", "agendar", "appointment", "book", "turno"]
    if any(s in t for s in booking_signals):
        return "booking"

    bye_signals = ["adiós", "chao", "bye", "hasta luego", "nada más", "eso es todo"]
    if any(s in t for s in bye_signals):
        return "end_call"

    return "collect_data"


def _compute_missing_fields(profile: ClientProfile, intent: str) -> list[str]:
    """
    Deterministic checklist: iterate REQUIRED_FIELDS, skip any field
    already present in client_profile.  This is the core of the entity
    resolution — if CRM returned the name, it physically cannot appear
    in the missing list.
    """
    missing = []
    for field in REQUIRED_FIELDS:
        if field == "intent":
            # "intent" is satisfied when user has expressed a clear purpose
            if intent in ("booking", "faq", "end_call"):
                continue
            # If we already know what they want, skip
            continue  # intent is auto-detected, never asked directly
        value = profile.profile_data.get(field)
        if not value:
            missing.append(field)
    return missing


# ===================================================================
# GRAPH NODES
# ===================================================================

async def init_system(state: dict) -> dict:
    """
    Initialization node — injects the system prompt exactly once.
    This prevents context pollution from prepending it on every webhook call.
    """
    if state.get("system_initialized"):
        return {}

    return {
        "messages": [SYSTEM_PROMPT],
        "system_initialized": True,
    }


async def universal_extract(state: dict) -> dict:
    """
    Universal Extraction Node — scans EVERY user input for ANY piece
    of profile information (DNI, name, email, phone).  Updates the
    client_profile.profile_data with anything found.

    This runs on every turn, regardless of intent, so the system
    captures data opportunistically (e.g. "Hola, soy María García
    y mi DNI es 12345678A" in a single utterance).
    """
    last_msg = _last_human_text(state)
    if not last_msg:
        return {}

    profile = _get_profile(state).model_copy(deep=True)

    # Scan for ALL extractable fields in every message
    extractable_fields = ["dni", "name", "email", "phone"]
    changed = False

    for field in extractable_fields:
        # Only extract if not already known
        if profile.profile_data.get(field):
            continue
        value = _extract_field_value(field, last_msg)
        if value:
            profile.profile_data[field] = value
            changed = True
            logger.info("Extracted %s=%s", field, value)

            # Set identity_key when we capture a unique identifier
            if field == "dni" and not profile.identity_key:
                profile.identity_key = value

    if changed:
        # Recompute missing fields
        intent = state.get("intent", "unknown")
        missing = _compute_missing_fields(profile, intent)
        return {
            "client_profile": profile,
            "missing_required_fields": missing,
        }
    return {}


async def crm_sync(state: dict) -> dict:
    """
    CRM Sync Node — the core of entity resolution.

    Trigger: Runs whenever a unique identifier (DNI) is first captured.
    Action:  Queries the CRM.
    Behavior:
      - If customer EXISTS: hydrates client_profile with ALL known data
        from the DB.  This instantly marks those fields as "known" and
        the deterministic checklist router will skip them.
      - If customer does NOT exist: marks is_new_customer = True.
    """
    profile = _get_profile(state)

    # Only trigger on first identifier capture, not on subsequent turns
    if profile.is_verified:
        return {}

    dni = profile.profile_data.get("dni")
    if not dni:
        return {}

    # Query CRM
    crm_record = await lookup_user(dni)

    if crm_record:
        # HYDRATE — fill client_profile with all CRM data
        hydrated = profile.model_copy(deep=True)
        hydrated.identity_key = dni
        hydrated.is_verified = True
        hydrated.is_new_customer = False

        # Merge CRM data into profile_data (CRM wins for existing fields)
        for key, value in crm_record.items():
            if key == "interaction_history":
                hydrated.interaction_history = value
            elif value:
                hydrated.profile_data[key] = value

        # Recompute missing fields — CRM data makes fields "known"
        missing = _compute_missing_fields(hydrated, state.get("intent", "unknown"))

        logger.info(
            "CRM HYDRATED: %s — fields filled from DB, missing=%s",
            crm_record.get("name", dni), missing,
        )
        return {
            "client_profile": hydrated,
            "missing_required_fields": missing,
        }
    else:
        # New customer — mark as such
        updated = profile.model_copy(deep=True)
        updated.is_new_customer = True
        logger.info("CRM MISS: DNI %s not found — new customer", dni)
        return {"client_profile": updated}


async def checklist_router(state: dict) -> dict:
    """
    Deterministic Checklist Router — classifies intent and ensures the
    missing_required_fields list is up to date.

    The router iterates through REQUIRED_FIELDS and skips any field
    already present in client_profile.profile_data.  This is what makes
    the system behave like a human clerk: if CRM returned the name,
    the bot physically cannot route to the "Ask Name" node.
    """
    last_msg = _last_human_text(state)
    intent = _detect_intent(last_msg) if last_msg else state.get("intent", "unknown")

    profile = _get_profile(state)
    booking = _get_booking(state)

    updates: dict[str, Any] = {"intent": intent}

    # Override intent when we're waiting for a booking response
    if booking.status == "offered" and intent == "collect_data":
        intent = "booking"
        updates["intent"] = intent

    # If we were in slot-filling and user asked a FAQ, flag it
    if intent == "faq" and state.get("current_slot"):
        updates["interrupted_by_faq"] = True

    # Recompute missing fields
    missing = _compute_missing_fields(profile, intent)
    updates["missing_required_fields"] = missing

    return updates


async def greet(state: dict) -> dict:
    """Generate a greeting — personalized if we know the client."""
    profile = _get_profile(state)
    name = profile.profile_data.get("name", "")

    if profile.is_verified and name:
        # Known customer — warm greeting with history context
        history = profile.interaction_history
        if history:
            last_interaction = history[-1]
            text = (
                f"¡Hola, {name}! Bienvenido/a de nuevo a Omvyx. "
                f"Veo que su última visita fue el {last_interaction.get('date', '')} "
                f"por {last_interaction.get('summary', 'una consulta')}. "
                "¿En qué puedo ayudarle hoy?"
            )
        else:
            text = f"¡Hola, {name}! Bienvenido/a de nuevo a Omvyx. ¿En qué puedo ayudarle hoy?"
    elif name:
        text = f"¡Hola, {name}! Bienvenido/a a Omvyx. ¿En qué puedo ayudarle hoy?"
    else:
        text = "¡Hola! Bienvenido/a a Omvyx. ¿En qué puedo ayudarle hoy?"

    return {"messages": [AIMessage(content=text)]}


async def handle_faq(state: dict) -> dict:
    """
    Answer a FAQ question.  If the user was in the middle of slot-filling
    (interrupted_by_faq=True), append a prompt to resume data collection.
    """
    last_msg = _last_human_text(state)

    answer = await search_faq(last_msg)
    if not answer:
        answer = "Lo siento, no tengo información sobre eso. ¿Puedo ayudarle con algo más?"

    # Resume slot-filling after answering the FAQ
    missing = state.get("missing_required_fields", [])
    if state.get("interrupted_by_faq") and missing:
        next_field = missing[0]
        answer += f" Pero volviendo a sus datos, {FIELD_PROMPTS[next_field].lower()}"

    return {
        "messages": [AIMessage(content=answer)],
        "interrupted_by_faq": False,
    }


async def collect_data(state: dict) -> dict:
    """
    Slot-filling node.  Checks missing_required_fields and asks for the next one.
    When all fields are captured, confirms completion.
    """
    missing = state.get("missing_required_fields", [])
    profile = _get_profile(state)

    # All captured — confirm
    if not missing:
        name = profile.profile_data.get("name", "")
        if profile.is_verified:
            text = (
                f"He encontrado su expediente, {name}. Ya tengo todos sus datos. "
                "¿Le gustaría reservar una cita o necesita algo más?"
            )
        else:
            text = (
                f"Perfecto, {name}. Ya tengo todos sus datos registrados. "
                "¿Le gustaría reservar una cita?"
            )
        return {
            "messages": [AIMessage(content=text)],
            "current_slot": "",
        }

    # Ask for the next missing field
    next_field = missing[0]
    prompt_text = FIELD_PROMPTS[next_field]

    return {
        "messages": [AIMessage(content=prompt_text)],
        "current_slot": next_field,
    }


async def manage_booking(state: dict) -> dict:
    """
    Handle appointment booking.  If user data isn't complete yet,
    redirect to data collection first.
    """
    missing = state.get("missing_required_fields", [])
    profile = _get_profile(state)

    # Need user data before booking
    if missing:
        text = (
            "Con gusto le ayudo a reservar una cita. "
            f"Pero primero necesito algunos datos. {FIELD_PROMPTS[missing[0]]}"
        )
        return {
            "messages": [AIMessage(content=text)],
            "intent": "collect_data",
            "current_slot": missing[0],
        }

    booking = _get_booking(state)

    # Extract date from last message if not yet set
    if not booking.requested_date:
        last_msg = _last_human_text(state)
        date_match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", last_msg)
        if date_match:
            booking = BookingRequest(
                requested_date=date_match.group(0), status="checking"
            )
        else:
            text = (
                "¿Para qué fecha y hora le gustaría la cita? "
                "Por ejemplo: 2026-02-11 10:00"
            )
            return {"messages": [AIMessage(content=text)]}

    # Check availability
    if booking.status == "checking":
        result = await check_availability(booking.requested_date)
        if result.get("available"):
            dni = profile.profile_data.get("dni", "")
            await book_slot(booking.requested_date, dni)
            booking = BookingRequest(
                requested_date=booking.requested_date,
                confirmed_slot=booking.requested_date,
                status="confirmed",
            )
            text = (
                f"¡Listo! Su cita ha quedado confirmada para el "
                f"{booking.confirmed_slot}. ¿Necesita algo más?"
            )
        else:
            alts = result.get("alternatives", [])
            booking = BookingRequest(
                requested_date=booking.requested_date, status="offered"
            )
            if alts:
                options = " o ".join(alts)
                text = (
                    f"Lo siento, el {booking.requested_date} no está disponible. "
                    f"Tengo disponibilidad el {options}. "
                    "¿Le viene bien alguna de estas opciones?"
                )
            else:
                error = result.get("error", "No hay disponibilidad.")
                text = f"Lo siento, {error} ¿Desea intentar otra fecha?"

        return {
            "messages": [AIMessage(content=text)],
            "booking": booking,
        }

    # If we offered alternatives and user responds
    if booking.status == "offered":
        last_msg = _last_human_text(state)
        date_match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", last_msg)
        if date_match:
            slot = date_match.group(0)
            result = await check_availability(slot)
            if result.get("available"):
                dni = profile.profile_data.get("dni", "")
                await book_slot(slot, dni)
                booking = BookingRequest(
                    requested_date=slot,
                    confirmed_slot=slot,
                    status="confirmed",
                )
                text = (
                    f"¡Perfecto! Cita confirmada para el {slot}. "
                    "¿Algo más en lo que pueda ayudarle?"
                )
            else:
                text = "Ese horario tampoco está disponible. ¿Quiere probar otra fecha?"
        else:
            booking = BookingRequest(status="idle")
            text = "Entendido. ¿Para qué fecha y hora le gustaría la cita?"

        return {
            "messages": [AIMessage(content=text)],
            "booking": booking,
        }

    # Fallback
    return {
        "messages": [AIMessage(content="¿Le gustaría reservar una cita? Dígame la fecha y hora deseada.")],
        "booking": BookingRequest(status="idle"),
    }


async def end_call(state: dict) -> dict:
    """Polite goodbye."""
    profile = _get_profile(state)
    name = profile.profile_data.get("name", "")
    if name:
        text = f"Ha sido un placer atenderle, {name}. ¡Hasta pronto!"
    else:
        text = "Gracias por llamar a Omvyx. ¡Hasta pronto!"
    return {"messages": [AIMessage(content=text)]}


async def save_crm(state: dict) -> dict:
    """
    Business Logic Guard — Duplicate Registration Prevention.

    If is_verified is True → UPDATE (client already exists in CRM).
    If is_new_customer is True → INSERT (new client).
    This runs before the final respond node to ensure CRM is in sync.
    """
    profile = _get_profile(state)
    dni = profile.profile_data.get("dni")

    # Nothing to save without a DNI
    if not dni:
        return {}

    # Skip if already verified and no new data to push
    if profile.is_verified:
        # UPDATE — push any newly collected fields back to CRM
        await update_user(
            dni,
            name=profile.profile_data.get("name", ""),
            email=profile.profile_data.get("email", ""),
            phone=profile.profile_data.get("phone", ""),
        )
        logger.info("CRM UPDATE: %s", dni)
    elif profile.is_new_customer and profile.profile_data.get("name"):
        # INSERT — only when we have at minimum DNI + name
        await create_user(
            name=profile.profile_data.get("name", ""),
            dni=dni,
            email=profile.profile_data.get("email", ""),
            phone=profile.profile_data.get("phone", ""),
        )
        # Mark as verified now that they're in the system
        updated = profile.model_copy(deep=True)
        updated.is_verified = True
        updated.is_new_customer = False
        logger.info("CRM INSERT: %s", dni)
        return {"client_profile": updated}

    return {}


async def respond(state: dict) -> dict:
    """
    Terminal passthrough — clean convergence point before END.
    The actual response message was already appended by the preceding node.
    """
    return {}


# ===================================================================
# ROUTING LOGIC
# ===================================================================

def route_after_init(state: dict) -> str:
    """Always proceed to universal extraction."""
    return "universal_extract"


def route_after_extract(state: dict) -> str:
    """Always proceed to CRM sync."""
    return "crm_sync"


def route_after_crm_sync(state: dict) -> str:
    """Always proceed to checklist router."""
    return "checklist_router"


def route_after_checklist(state: dict) -> str:
    """Conditional edge from checklist_router → next node."""
    intent = state.get("intent", "unknown")
    missing = state.get("missing_required_fields", [])

    # If slot-filling is active and user didn't ask a FAQ or booking,
    # stay in collect_data regardless of what was detected
    if missing and intent not in ("faq", "booking", "end_call", "greeting"):
        return "collect_data"
    return intent


def route_after_collect(state: dict) -> str:
    """After collect_data, go to save_crm."""
    return "save_crm"


def route_after_booking(state: dict) -> str:
    """After manage_booking, redirect to collect_data if intent was changed."""
    if state.get("intent") == "collect_data":
        return "collect_data"
    return "save_crm"


def route_after_save(state: dict) -> str:
    """After save_crm, always go to respond."""
    return "respond"


# ===================================================================
# GRAPH ASSEMBLY
# ===================================================================

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph (uncompiled).
    """
    builder = StateGraph(OmvyxState)

    # --- Nodes ---
    builder.add_node("init_system", init_system)
    builder.add_node("universal_extract", universal_extract)
    builder.add_node("crm_sync", crm_sync)
    builder.add_node("checklist_router", checklist_router)
    builder.add_node("greet", greet)
    builder.add_node("handle_faq", handle_faq)
    builder.add_node("collect_data", collect_data)
    builder.add_node("manage_booking", manage_booking)
    builder.add_node("end_call", end_call)
    builder.add_node("save_crm", save_crm)
    builder.add_node("respond", respond)

    # --- Entry ---
    builder.set_entry_point("init_system")

    # --- Pipeline: init → extract → crm_sync → checklist_router ---
    builder.add_edge("init_system", "universal_extract")
    builder.add_edge("universal_extract", "crm_sync")
    builder.add_edge("crm_sync", "checklist_router")

    # --- Conditional edges from checklist_router ---
    builder.add_conditional_edges(
        "checklist_router",
        route_after_checklist,
        {
            "greeting": "greet",
            "faq": "handle_faq",
            "collect_data": "collect_data",
            "booking": "manage_booking",
            "end_call": "end_call",
            "unknown": "collect_data",
        },
    )

    # --- Edges from action nodes → save_crm → respond → END ---
    builder.add_edge("greet", "save_crm")
    builder.add_edge("handle_faq", "save_crm")
    builder.add_edge("end_call", "save_crm")

    builder.add_conditional_edges(
        "collect_data",
        route_after_collect,
        {"save_crm": "save_crm"},
    )

    builder.add_conditional_edges(
        "manage_booking",
        route_after_booking,
        {
            "save_crm": "save_crm",
            "collect_data": "collect_data",
        },
    )

    builder.add_edge("save_crm", "respond")
    builder.add_edge("respond", END)

    return builder


def compile_graph(checkpointer=None):
    """
    Build and compile the graph with checkpointing.

    Args:
        checkpointer: A LangGraph checkpointer instance.  Defaults to
                      MemorySaver (in-memory, fine for dev/single-process).
                      For production, pass AsyncSqliteSaver or a Redis-backed
                      checkpointer.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)
