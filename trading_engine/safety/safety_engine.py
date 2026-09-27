"""Safety Engine (README §50.2) : NORMAL / DEGRADED / HALTED.

Supervise données, modèles, risque et contrôles d'ordres, indépendamment des
modèles de signal.

- DEGRADED : qualité des données en baisse, symbole figé ou probable
  corporate action (symbole gelé), dérive d'un modèle. Effet : rebalancements
  réduits, symboles concernés gelés à leur poids courant.
- HALTED : données inutilisables, explosion numérique d'un modèle, perte
  journalière maximale, dépassement de risque critique, rejets répétés des
  hard controls. Effet : aucune nouvelle décision (seul UDONOTHING).

L'escalade est immédiate. Le retour DEGRADED -> NORMAL demande
`recovery_intervals` évaluations saines consécutives. HALTED ne se lève que
par `reset()` manuel, et ne déclenche **aucune liquidation**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Iterable

logger = logging.getLogger(__name__)


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"


_RANK = {SafetyState.NORMAL: 0, SafetyState.DEGRADED: 1, SafetyState.HALTED: 2}


@dataclass(frozen=True)
class SafetyConfig:
    degrade_data_score: float = 0.90       # score d'intégrité en dessous -> DEGRADED
    halt_data_score: float = 0.50          # score global en dessous -> HALTED
    max_daily_loss: float = 0.03           # perte depuis l'ouverture du jour UTC -> HALTED
    halt_on_breaches: tuple[str, ...] = ("DRAWDOWN", "LEVERAGE")
    max_hard_control_rejections: int = 3   # par jour -> HALTED
    recovery_intervals: int = 3
    corporate_action_hold: float = 86400.0  # secondes de gel après une corporate action probable
    degraded_rebalance_factor: float = 0.5  # fraction du chemin vers la cible en DEGRADED


@dataclass(frozen=True)
class SafetyStatus:
    state: SafetyState
    reasons: tuple[str, ...]
    frozen_symbols: frozenset[str]
    timestamp: datetime | None

    @property
    def can_decide(self) -> bool:
        return self.state is not SafetyState.HALTED


@dataclass
class ModelHealth:
    drift_events: int = 0            # dérives détectées depuis la dernière évaluation
    non_finite: bool = False         # coefficients / probabilités non finis
    details: list[str] = field(default_factory=list)


class SafetyEngine:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.state = SafetyState.NORMAL
        self.status = SafetyStatus(SafetyState.NORMAL, (), frozenset(), None)
        self.history: list[SafetyStatus] = []
        self._healthy_streak = 0
        self._day: date | None = None
        self._day_open_value: float | None = None
        self._last_value: float | None = None
        self._rejections: dict[date, int] = {}
        self._halt_reasons: tuple[str, ...] = ()

    # -------------------------------------------------------------- entrées

    def on_value(self, value: float, timestamp: datetime) -> None:
        if timestamp.date() != self._day:
            self._day = timestamp.date()
            self._day_open_value = value
        self._last_value = value

    @property
    def daily_return(self) -> float | None:
        if not self._day_open_value or self._last_value is None:
            return None
        return self._last_value / self._day_open_value - 1.0

    def record_hard_control_rejection(self, timestamp: datetime, reason: str) -> None:
        day = timestamp.date()
        self._rejections[day] = self._rejections.get(day, 0) + 1
        logger.warning("hard control rejection: %s", reason)

    # -------------------------------------------------------------- évaluation

    def evaluate(
        self,
        timestamp: datetime | None,
        *,
        data_scores: dict[str, float] | None = None,
        stale_symbols: Iterable[str] = (),
        corporate_actions: dict[str, datetime] | None = None,
        risk_breaches: Iterable[str] = (),
        model_health: ModelHealth | None = None,
    ) -> SafetyStatus:
        cfg = self.config
        halt: list[str] = []
        degrade: list[str] = []
        frozen: set[str] = set()
        data_scores = data_scores or {}
        model_health = model_health or ModelHealth()

        # données
        if data_scores and min(data_scores.values()) < cfg.halt_data_score:
            halt.append(f"DATA_UNUSABLE min_score={min(data_scores.values()):.2f}")
        for sym, score in sorted(data_scores.items()):
            if score < cfg.degrade_data_score:
                degrade.append(f"DATA_QUALITY {sym}={score:.2f}")
                frozen.add(sym)
        for sym in stale_symbols:
            degrade.append(f"STALE_FEED {sym}")
            frozen.add(sym)
        for sym, when in sorted((corporate_actions or {}).items()):
            if timestamp is None or timestamp - when <= timedelta(seconds=cfg.corporate_action_hold):
                degrade.append(f"CORPORATE_ACTION {sym}")
                frozen.add(sym)

        # modèles
        if model_health.non_finite:
            halt.append("MODEL_EXPLOSION " + "; ".join(model_health.details))
        elif model_health.drift_events:
            degrade.append(f"MODEL_DRIFT x{model_health.drift_events}")

        # risque et pertes
        for kind in risk_breaches:
            if kind in cfg.halt_on_breaches:
                halt.append(f"RISK_BREACH {kind}")
        daily = self.daily_return
        if daily is not None and -daily > cfg.max_daily_loss:
            halt.append(f"DAILY_LOSS {daily:.2%}")

        # contrôles d'ordres
        if timestamp is not None:
            rejections = self._rejections.get(timestamp.date(), 0)
            if rejections >= cfg.max_hard_control_rejections:
                halt.append(f"HARD_CONTROL_REJECTIONS x{rejections}")

        new_state = self._transition(halt, degrade)
        if new_state is SafetyState.HALTED:
            if halt:
                self._halt_reasons = tuple(halt)
            reasons = self._halt_reasons     # reste figé jusqu'au reset manuel
        elif new_state is SafetyState.DEGRADED:
            reasons = tuple(degrade) or (f"RECOVERING {self._healthy_streak}/{cfg.recovery_intervals}",)
        else:
            reasons = ()
        self.status = SafetyStatus(new_state, reasons, frozenset(frozen), timestamp)
        if not self.history or self.history[-1].state is not new_state:
            logger.warning("safety state -> %s (%s)", new_state.value, "; ".join(reasons) or "ok")
            self.history.append(self.status)
        return self.status

    def _transition(self, halt: list[str], degrade: list[str]) -> SafetyState:
        if self.state is SafetyState.HALTED or halt:
            self.state = SafetyState.HALTED
            return self.state
        if degrade:
            self.state = SafetyState.DEGRADED
            self._healthy_streak = 0
            return self.state
        if self.state is SafetyState.DEGRADED:
            self._healthy_streak += 1
            if self._healthy_streak >= self.config.recovery_intervals:
                self.state = SafetyState.NORMAL
        return self.state

    def reset(self, operator: str, timestamp: datetime | None = None) -> SafetyStatus:
        """Remise en route manuelle après HALTED (repasse par DEGRADED)."""
        if self.state is not SafetyState.HALTED:
            return self.status
        logger.warning("safety reset by %s", operator)
        self.state = SafetyState.DEGRADED
        self._healthy_streak = 0
        self._halt_reasons = ()
        if timestamp is not None:
            self._rejections.pop(timestamp.date(), None)
        self.status = SafetyStatus(SafetyState.DEGRADED, (f"RESET by {operator}",), frozenset(), timestamp)
        self.history.append(self.status)
        return self.status
