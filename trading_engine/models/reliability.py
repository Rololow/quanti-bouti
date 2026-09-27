"""Fiabilité d'un modèle par contexte (README §50.5).

Prédictibilité ≠ fiabilité : un modèle peut prédire +2 % avec une faible
incertitude et avoir été peu fiable récemment **dans ce type de marché**.

Pour chaque contexte (ex. régime HF du symbole), on suit le skill hors
échantillon (1 - MSE / MSE de la prévision nulle). Comme les contextes rares
ont peu de résultats, le skill de contexte est ramené vers le skill global :

    skill_ctx* = n / (n + k) * skill_ctx + k / (n + k) * skill_global

La fiabilité exposée est max(skill*, 0) : 0 = le modèle ne bat pas « prédire
zéro » dans ce contexte, et ne doit pas influencer la cible.
"""

from __future__ import annotations

from trading_engine.models.monitoring import ModelMonitor

GLOBAL = "ALL"


class ContextReliability:
    def __init__(self, lam: float = 0.98, prior_strength: float = 50.0) -> None:
        self.lam = lam
        self.prior_strength = prior_strength
        self._monitors: dict[str, ModelMonitor] = {GLOBAL: ModelMonitor(lam)}

    def update(self, context: str, realized: float, mean: float, std: float) -> None:
        self._monitors[GLOBAL].update(realized, mean, std)
        if context != GLOBAL:
            self._monitors.setdefault(context, ModelMonitor(self.lam)).update(realized, mean, std)

    def skill(self, context: str = GLOBAL) -> float | None:
        global_skill = self._monitors[GLOBAL].skill
        monitor = self._monitors.get(context)
        if context == GLOBAL or monitor is None or monitor.skill is None:
            return global_skill
        if global_skill is None:
            return monitor.skill
        w = monitor.n / (monitor.n + self.prior_strength)
        return w * monitor.skill + (1 - w) * global_skill

    def reliability(self, context: str = GLOBAL) -> float | None:
        skill = self.skill(context)
        return None if skill is None else max(0.0, min(skill, 1.0))

    def contexts(self) -> dict[str, tuple[int, float | None]]:
        """contexte -> (nombre de résultats, skill ramené)."""
        return {ctx: (m.n, self.skill(ctx)) for ctx, m in sorted(self._monitors.items())}
