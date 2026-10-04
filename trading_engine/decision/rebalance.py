"""Decision Engine : UREBALANCE / UDONOTHING (README §31, §49.6-49.9, §50.8).

Il n'y a ni BUY ni SELL : la seule question est « rapprocher le portefeuille
de sa cible vaut-il ce que ça coûte, compte tenu de l'incertitude ? ».

Tout est exprimé en unités économiques (devise du portefeuille, V = valeur) :

    TE²(w)          = (w - w_target)' Σ (w - w_target)          (Σ annualisée)
    risque(f)       = V · γ/2 · H · [TE²(w_courant) - TE²(w_f)]  (H = période de détention)
    alpha(f)        = V · Σ_i (w_f,i - w_c,i) · μ_i · fiabilité_i
    coûts(f)        = spread + slippage + commissions + taxes (TOB, plus-values)
    σ(f)            = sqrt(σ_alpha² + ((1 - robustesse) · risque(f))² + σ_exec²)
    σ_exec          = coûts d'exécution · (1 - confiance d'exécution)
    σ_alpha         = V · sqrt(Σ_i ((w_f,i - w_c,i) · se_i)²)
    se_i            = std_i / sqrt(n_obs_i) + désaccord_i

σ_alpha mesure l'incertitude sur l'**estimation** de μ (erreur-type + désaccord
des modèles), pas le bruit du rendement réalisé : ce bruit est le risque de
marché, déjà valorisé par le terme de risque. Le compter deux fois écraserait
toute décision.

σ_exec traduit la séparation MODEL ≠ EXECUTION CONFIDENCE (README §49.8) :
tant que les modèles d'exécution reposent sur des hypothèses (confiance 0),
leur coût estimé compte double ; il n'est pas ajouté si la confiance
d'exécution n'est pas fournie.
    net(f)          = risque(f) + alpha(f) - coûts(f) - k · σ(f)

avec w_f = w_c + f · (w_target - w_c) pour f dans `fractions` (rebalancement
partiel). UREBALANCE à la meilleure fraction si net > `min_net_benefit`,
sinon UDONOTHING.

Filtres (avant tout calcul) : HALTED, pas de cible, cible non robuste,
symboles gelés ou aux données dégradées, bande de non-trading
|w_target - w_courant| < ε (hystérésis).

Les diagnostics (accord des modèles, fiabilité, qualité des données,
robustesse, risque) sont gardés pour l'explication mais **ne sont pas
multipliés** entre eux : chacun agit via un filtre ou via σ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from trading_engine.execution.cost_model import ExecutionCostModel
from trading_engine.signals.signal import Signal
from trading_engine.tax.tax_model import TaxModel
from trading_engine.timeutils import periods_per_year

UREBALANCE = "UREBALANCE"
UDONOTHING = "UDONOTHING"


@dataclass(frozen=True)
class DecisionConfig:
    enabled: bool = True
    risk_aversion: float = 5.0            # γ
    holding_period: str = "5d"            # H : horizon sur lequel l'écart de risque est valorisé
    k_sigma: float = 1.0                  # exigence de marge par rapport à l'incertitude
    no_trade_band: float = 0.02           # hystérésis |w_target - w_courant|
    fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    min_net_benefit: float = 0.0
    default_spread_bps: float = 5.0       # si aucune quote n'est disponible
    slippage_bps: float = 2.0
    commission_per_order: float = 0.0
    min_robustness: float = 0.3           # en dessous : cible jugée non fiable
    min_data_score: float = 0.9
    include_alpha: bool = True
    # Exposition brute courante > plafond + cette tolérance (dérive des prix
    # avec levier) : désendettement forcé vers la cible, sans bande ni calcul
    # coût/bénéfice (le risque de marge n'est pas dans ce calcul).
    delever_tolerance: float = 0.05


@dataclass(frozen=True)
class SymbolDecision:
    symbol: str
    current_weight: float
    target_weight: float
    execution_weight: float
    notional: float                       # signé : > 0 achat
    excluded: str | None = None           # raison d'exclusion (gel, bande, données…)


@dataclass(frozen=True)
class CostBreakdown:
    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    transaction_tax: float = 0.0
    capital_gains_tax: float = 0.0
    impact: float = 0.0

    @property
    def execution(self) -> float:
        """Coûts qui dépendent de la façon d'exécuter (hors taxes)."""
        return self.spread + self.slippage + self.impact + self.commission

    @property
    def total(self) -> float:
        return self.execution + self.transaction_tax + self.capital_gains_tax


@dataclass(frozen=True)
class Decision:
    decision_id: str
    timestamp: datetime | None
    action: str
    fraction: float                       # part du chemin vers la cible (0 si UDONOTHING)
    symbols: dict[str, SymbolDecision]
    # Fraction à laquelle se rapportent les montants économiques ci-dessous :
    # pour un UDONOTHING, c'est la meilleure option évaluée (et rejetée).
    evaluated_fraction: float = 0.0
    # économie (devise du portefeuille)
    risk_benefit: float = 0.0
    alpha_benefit: float = 0.0
    costs: CostBreakdown = field(default_factory=CostBreakdown)
    uncertainty: float = 0.0
    net_benefit: float = 0.0
    u_donothing: float = 0.0              # coût de l'écart de risque si on ne fait rien
    urgency: float = 0.0                  # part du bénéfice qui s'évapore (alpha) vs durable (risque)
    # diagnostics (explication, pas multipliés)
    tracking_error: float | None = None   # annualisée, avant décision
    model_agreement: float | None = None  # modèles effectifs / modèles
    model_reliability: float | None = None
    data_quality: float | None = None
    robustness: float | None = None
    portfolio_risk: float | None = None
    safety_state: str | None = None
    execution_confidence: float | None = None   # modèle d'exécution : phase suivante
    reasons: tuple[str, ...] = ()

    @property
    def execution_weights(self) -> dict[str, float]:
        return {s: d.execution_weight for s, d in self.symbols.items()}


@dataclass
class DecisionContext:
    """Tout ce dont la décision a besoin, fourni par le moteur."""

    timestamp: datetime | None
    portfolio_value: float
    current: Mapping[str, float]
    target: Mapping[str, float] | None
    symbols: Sequence[str]
    cov: np.ndarray | None                # annualisée, alignée sur `symbols`
    prices: Mapping[str, float]
    spreads: Mapping[str, float] = field(default_factory=dict)   # spread relatif (ask-bid)/mid
    signals: Mapping[str, Signal | None] = field(default_factory=dict)
    safety_state: str = "NORMAL"
    market_block: str | None = None       # marché fermé / ouverture / clôture : pas de trade
    frozen: frozenset[str] = frozenset()
    data_scores: Mapping[str, float] = field(default_factory=dict)
    robustness: float | None = None
    portfolio_risk: float | None = None
    model_agreement: float | None = None
    daily_vols: Mapping[str, float] = field(default_factory=dict)
    adv: Mapping[str, float | None] = field(default_factory=dict)
    execution_confidence: float | None = None
    max_gross: float | None = None        # plafond d'exposition brute (désendettement forcé)


class DecisionEngine:
    def __init__(
        self,
        config: DecisionConfig | None = None,
        tax: TaxModel | None = None,
        cost_model: ExecutionCostModel | None = None,
    ) -> None:
        self.config = config or DecisionConfig()
        self.tax = tax
        self.cost_model = cost_model
        self._horizon_years = 1.0 / periods_per_year(self.config.holding_period)
        self._counter = 0

    def _new_id(self, ts: datetime | None) -> str:
        self._counter += 1
        stamp = "na" if ts is None else ts.strftime("%Y%m%dT%H%M%S")
        return f"D{stamp}-{self._counter:05d}"

    # ------------------------------------------------------------------ utilitaires

    def _te2(self, w: np.ndarray, target: np.ndarray, cov: np.ndarray) -> float:
        d = w - target
        return max(float(d @ cov @ d), 0.0)

    def _costs(self, ctx: DecisionContext, trades: Mapping[str, float]) -> CostBreakdown:
        cfg = self.config
        spread = slippage = commission = impact = 0.0
        for sym, notional in trades.items():
            if abs(notional) < 1e-9:
                continue
            price = ctx.prices.get(sym)
            if self.cost_model is not None and price:
                # Même modèle que l'exécution ; hypothèse prudente : l'ordre croise le spread.
                est = self.cost_model.estimate(
                    notional / price, price, rel_spread=ctx.spreads.get(sym), crossing=1.0,
                    daily_vol=ctx.daily_vols.get(sym), adv=ctx.adv.get(sym),
                )
                spread += est.spread
                slippage += est.slippage
                impact += est.impact
                commission += est.fees
                continue
            rel_spread = ctx.spreads.get(sym, cfg.default_spread_bps * 1e-4)
            spread += abs(notional) * rel_spread / 2.0
            slippage += abs(notional) * cfg.slippage_bps * 1e-4
            commission += cfg.commission_per_order
        tx = cg = 0.0
        if self.tax is not None and ctx.timestamp is not None and trades:
            tax_cost = self.tax.rebalance_cost(trades, ctx.prices, ctx.timestamp)
            tx, cg = tax_cost.transaction_tax, tax_cost.capital_gains_tax
        return CostBreakdown(spread, slippage, commission, tx, cg, impact)

    def _nothing(
        self, ctx: DecisionContext, reasons: list[str],
        excluded: Mapping[str, str] | None = None, **diagnostics,
    ) -> Decision:
        excluded = excluded or {}
        target = ctx.target or {}
        symbols = {
            s: SymbolDecision(s, ctx.current.get(s, 0.0), target.get(s, ctx.current.get(s, 0.0)),
                              ctx.current.get(s, 0.0), 0.0, excluded.get(s))
            for s in ctx.symbols
        }
        return Decision(
            decision_id=self._new_id(ctx.timestamp), timestamp=ctx.timestamp, action=UDONOTHING,
            fraction=0.0, symbols=symbols, reasons=tuple(reasons),
            safety_state=ctx.safety_state, robustness=ctx.robustness,
            portfolio_risk=ctx.portfolio_risk, model_agreement=ctx.model_agreement,
            data_quality=min(ctx.data_scores.values(), default=None), **diagnostics,
        )

    # ------------------------------------------------------------------ décision

    def decide(self, ctx: DecisionContext) -> Decision:
        cfg = self.config
        # --- filtres
        if ctx.safety_state == "HALTED":
            return self._nothing(ctx, ["safety HALTED : aucune nouvelle décision"])
        if ctx.market_block:
            return self._nothing(ctx, [ctx.market_block])
        if not ctx.target:
            return self._nothing(ctx, ["pas encore de cible"])
        if ctx.cov is None or ctx.portfolio_value <= 0:
            return self._nothing(ctx, ["risque non estimable (covariance ou valeur manquante)"])
        if ctx.robustness is not None and ctx.robustness < cfg.min_robustness:
            return self._nothing(ctx, [f"cible non robuste (score {ctx.robustness:.2f} < {cfg.min_robustness:.2f})"])

        symbols = list(ctx.symbols)
        V = ctx.portfolio_value
        w_c = np.array([ctx.current.get(s, 0.0) for s in symbols])
        w_t_full = np.array([ctx.target.get(s, ctx.current.get(s, 0.0)) for s in symbols])

        gross_c = float(np.abs(w_c).sum())
        delever = (ctx.max_gross is not None and gross_c > ctx.max_gross + cfg.delever_tolerance
                   and float(np.abs(w_t_full).sum()) < gross_c)

        # Symboles exclus : ils restent à leur poids courant.
        excluded: dict[str, str] = {}
        for i, s in enumerate(symbols):
            if s in ctx.frozen:
                excluded[s] = "gelé par le Safety Engine"
            elif ctx.data_scores.get(s, 1.0) < cfg.min_data_score:
                excluded[s] = f"qualité des données {ctx.data_scores[s]:.2f}"
            elif delever:
                continue
            elif abs(w_t_full[i] - w_c[i]) < cfg.no_trade_band:
                excluded[s] = f"écart {w_t_full[i] - w_c[i]:+.1%} dans la bande de non-trading"
        movable = np.array([s not in excluded for s in symbols])
        w_t = np.where(movable, w_t_full, w_c)

        te2_now = self._te2(w_c, w_t_full, ctx.cov)
        tracking_error = math.sqrt(te2_now)
        scale = V * cfg.risk_aversion / 2.0 * self._horizon_years
        u_donothing = -scale * te2_now

        mu = np.zeros(len(symbols))
        sd = np.zeros(len(symbols))
        reliabilities = []
        for i, s in enumerate(symbols):
            sig = ctx.signals.get(s)
            if sig is None or not cfg.include_alpha:
                continue
            rel = sig.reliability or 0.0
            mu[i] = sig.mean * rel
            # incertitude d'estimation de μ (pas le bruit du rendement)
            sd[i] = sig.std / math.sqrt(max(sig.n_obs, 1)) + (sig.disagreement or 0.0)
            reliabilities.append(rel)

        diag = dict(
            tracking_error=tracking_error, u_donothing=u_donothing,
            model_reliability=float(np.mean(reliabilities)) if reliabilities else None,
        )
        if not movable.any() or np.allclose(w_t, w_c):
            reasons = [f"aucun écart hors bande (TE vers la cible {tracking_error:.2%})"]
            reasons += [f"{s}: {why}" for s, why in excluded.items()]
            return self._nothing(ctx, reasons, excluded, **diag)

        # --- évaluation de chaque fraction
        robustness = 1.0 if ctx.robustness is None else ctx.robustness
        best = None
        for f in ((1.0,) if delever else sorted(set(cfg.fractions))):
            w_f = w_c + f * (w_t - w_c)
            dw = w_f - w_c
            risk_benefit = scale * (te2_now - self._te2(w_f, w_t_full, ctx.cov))
            alpha = V * float(dw @ mu)
            sigma_alpha = V * math.sqrt(float(np.sum((dw * sd) ** 2)))
            sigma = math.sqrt(sigma_alpha**2 + ((1 - robustness) * risk_benefit) ** 2)
            trades = {s: float(dw[i] * V) for i, s in enumerate(symbols) if abs(dw[i]) > 1e-12}
            costs = self._costs(ctx, trades)
            if ctx.execution_confidence is not None:
                sigma_exec = costs.execution * (1.0 - ctx.execution_confidence)
                sigma = math.sqrt(sigma**2 + sigma_exec**2)
            net = risk_benefit + alpha - costs.total - cfg.k_sigma * sigma
            candidate = (net, f, w_f, risk_benefit, alpha, costs, sigma, trades)
            if best is None or net > best[0]:
                best = candidate

        net, f, w_f, risk_benefit, alpha, costs, sigma, trades = best
        evaluated = f
        per_symbol = {
            s: SymbolDecision(s, float(w_c[i]), float(w_t_full[i]), float(w_f[i]),
                              trades.get(s, 0.0), excluded.get(s))
            for i, s in enumerate(symbols)
        }
        gross_benefit = max(risk_benefit, 0.0) + max(alpha, 0.0)
        urgency = max(alpha, 0.0) / gross_benefit if gross_benefit > 0 else 0.0
        reasons = self._reasons(per_symbol, risk_benefit, alpha, costs, sigma, net, f, tracking_error)
        action = UREBALANCE if net > cfg.min_net_benefit or delever else UDONOTHING
        if delever:
            reasons.insert(0, f"désendettement forcé : exposition brute {gross_c:.2f} > plafond {ctx.max_gross:.2f}")
        if action == UDONOTHING:
            reasons.insert(0, "le bénéfice ne couvre pas coûts + incertitude")
            f, per_symbol = 0.0, {
                s: SymbolDecision(d.symbol, d.current_weight, d.target_weight, d.current_weight, 0.0, d.excluded)
                for s, d in per_symbol.items()
            }
        return Decision(
            decision_id=self._new_id(ctx.timestamp), timestamp=ctx.timestamp, action=action,
            fraction=f, symbols=per_symbol, evaluated_fraction=evaluated,
            risk_benefit=risk_benefit, alpha_benefit=alpha,
            costs=costs, uncertainty=sigma, net_benefit=net, urgency=urgency,
            model_agreement=ctx.model_agreement,
            data_quality=min(ctx.data_scores.values(), default=None),
            robustness=ctx.robustness, portfolio_risk=ctx.portfolio_risk,
            safety_state=ctx.safety_state, reasons=tuple(reasons),
            execution_confidence=ctx.execution_confidence, **diag,
        )

    @staticmethod
    def _reasons(per_symbol, risk_benefit, alpha, costs, sigma, net, f, te) -> list[str]:
        moves = sorted(
            (d for d in per_symbol.values() if d.excluded is None and abs(d.target_weight - d.current_weight) > 0),
            key=lambda d: -abs(d.target_weight - d.current_weight),
        )
        reasons = [
            f"écart de risque à la cible (TE {te:.2%}) : bénéfice {risk_benefit:+.2f}",
            f"alpha attendu (pondéré par la fiabilité) : {alpha:+.2f}",
            f"coûts {costs.total:.2f} (spread {costs.spread:.2f}, slippage {costs.slippage:.2f}, "
            f"impact {costs.impact:.2f}, taxes {costs.transaction_tax + costs.capital_gains_tax:.2f})",
            f"incertitude {sigma:.2f} ; net {net:+.2f} à {f:.0%} du chemin",
        ]
        reasons += [
            f"{d.symbol}: {d.current_weight:.1%} → cible {d.target_weight:.1%}" for d in moves[:3]
        ]
        reasons += [f"{d.symbol}: {d.excluded}" for d in per_symbol.values() if d.excluded]
        return reasons
