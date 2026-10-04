"""Prix limite selon l'agressivité (README §33).

    achat : limite = bid + a · (ask - bid)      a = 0 passif (bid), a = 1 croise (ask)
    vente : limite = ask - a · (ask - bid)

Arrondi au tick, dans le sens prudent (achat vers le bas, vente vers le haut),
sauf pour un ordre qui doit croiser (a >= 1) : arrondi pour rester exécutable.
"""

from __future__ import annotations

import math


def limit_price(side: str, bid: float, ask: float, aggressiveness: float, tick: float = 0.01) -> float:
    if bid <= 0 or ask < bid:
        raise ValueError(f"invalid quote bid={bid} ask={ask}")
    a = max(0.0, min(aggressiveness, 1.0))
    spread = ask - bid
    if side == "buy":
        raw = bid + a * spread
        return (math.ceil(raw / tick - 1e-9) if a >= 1 else math.floor(raw / tick + 1e-9)) * tick
    if side == "sell":
        raw = ask - a * spread
        return (math.floor(raw / tick + 1e-9) if a >= 1 else math.ceil(raw / tick - 1e-9)) * tick
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def distance_to_execution(side: str, limit: float, bid: float, ask: float) -> float:
    """Écart relatif que le marché doit parcourir pour exécuter l'ordre
    (<= 0 : exécutable immédiatement)."""
    if side == "buy":
        return (ask - limit) / ask
    return (limit - bid) / bid
