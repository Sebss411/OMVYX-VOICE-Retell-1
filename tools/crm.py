"""
Omvyx Voice — CRM Tool (Mock)

Simulates a CRM backend with rich client profiles, interaction history,
loyalty levels, and support for both INSERT and UPDATE operations.

Replace the mock dictionaries with real DB/API calls in production.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Mock database — keyed by DNI
# ---------------------------------------------------------------------------

_MOCK_USERS: dict[str, dict[str, Any]] = {
    "12345678A": {
        "name": "María García",
        "dni": "12345678A",
        "email": "maria@example.com",
        "phone": "+34600111222",
        "address": "Calle Serrano 45, Madrid",
        "loyalty_level": "gold",
        "interaction_history": [
            {"type": "appointment", "date": "2025-12-10", "summary": "Consulta general"},
            {"type": "call", "date": "2026-01-15", "summary": "Cambio de cita"},
        ],
    },
    "87654321B": {
        "name": "Carlos López",
        "dni": "87654321B",
        "email": "carlos@example.com",
        "phone": "+34600333444",
        "address": "Avenida de la Constitución 12, Sevilla",
        "loyalty_level": "silver",
        "interaction_history": [
            {"type": "appointment", "date": "2026-01-20", "summary": "Primera visita"},
        ],
    },
}


async def lookup_user(dni: str) -> dict[str, Any] | None:
    """
    Search for an existing user by DNI.

    Returns the full user record (including interaction_history and
    loyalty_level) or None if not found.
    """
    return _MOCK_USERS.get(dni.strip().upper())


async def create_user(
    name: str, dni: str, email: str = "", phone: str = "", **extra: Any,
) -> dict[str, Any]:
    """Register a new user (INSERT) and return the created record."""
    record: dict[str, Any] = {
        "name": name.strip(),
        "dni": dni.strip().upper(),
        "email": email.strip().lower() if email else "",
        "phone": phone.strip() if phone else "",
        "address": extra.get("address", ""),
        "loyalty_level": "standard",
        "interaction_history": [],
    }
    _MOCK_USERS[record["dni"]] = record
    return record


async def update_user(dni: str, **fields: Any) -> dict[str, Any] | None:
    """
    Update an existing user's profile (UPDATE, not INSERT).

    Only overwrites fields that are explicitly passed and non-empty.
    Returns the updated record, or None if the user doesn't exist.
    """
    key = dni.strip().upper()
    record = _MOCK_USERS.get(key)
    if record is None:
        return None

    for field, value in fields.items():
        if value is not None and value != "":
            record[field] = value

    _MOCK_USERS[key] = record
    return record
