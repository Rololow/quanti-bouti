"""Simulateur rapide du profil : mêmes règles que le moteur, sans le moteur.

Le moteur rejoue chaque barre en événements (bus, journal, intégrité, modèles) :
une vingtaine d'années prend ~100 s. Ce module rejoue le même pipeline sur des
clôtures quotidiennes, en quelques secondes, pour pouvoir simuler des milliers
de trajectoires (bootstrap) ou balayer des réglages.

Ce qui est repris tel quel du moteur (mêmes classes) : allocation par budgets
de risque, contraintes, hard controls sur la cible, filtre de tendance, suivi
du drawdown, décision coût/bénéfice (`DecisionEngine`), modèle fiscal
(`TaxModel`), intérêts du cash et du levier.

Ce qui est recalculé en lot (numpy) parce que cela ne dépend que des prix :
covariances EWMA, poids de parité de risque, score de robustesse, bande
d'exposition, score de tendance.

Ce qui est approché :

- exécution à la clôture du jour de la décision, au prix de clôture, coût =
  demi-spread + slippage (le moteur exécute en cours de séance sur des trades
  synthétiques, au prix limite) ;
- pas de Safety Engine (arrêts, mode dégradé), pas de contrôle de
  participation au volume ;
- méthodes d'allocation `risk_parity` et `class_parity` seulement (pas de
  signaux, pas de HRP).

`validate` mesure l'écart avec le moteur sur la même trajectoire : toute
conclusion tirée du simulateur vaut à cet écart près.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np

from trading_engine.allocation.allocator import RiskAllocator
from trading_engine.allocation.constraints import ConstraintEngine, gross_cap_of
from trading_engine.allocation.risk_parity import risk_budget_weights
from trading_engine.allocation.trend import TrendFilter
from trading_engine.backtest import metrics
from trading_engine.backtest.benchmarks import StrategyResult, daily_closes
from trading_engine.backtest.dataset import meta_path, rates_path, read_closes
from trading_engine.config import Config
from trading_engine.decision.rebalance import DecisionContext, DecisionEngine
from trading_engine.execution.cost_model import ExecutionCostModel
from trading_engine.execution.hard_controls import HardControls
from trading_engine.portfolio.financing import RateSeries, accrue
from trading_engine.risk.portfolio_risk import DecayingPeakDrawdown, DrawdownTracker, RollingDrawdown
from trading_engine.tax.profile import load_tax_profile
from trading_engine.tax.tax_model import TaxModel, as_if_current_rules
from trading_engine.config import resolve_path
from trading_engine.tax.fx import FxRates
from trading_engine.timeutils import periods_per_year

UTC = timezone.utc
# Le moteur finalise les rendements d'une séance quand arrive la barre de la
# séance suivante, elle-même close au premier trade du lendemain : la
# covariance d'une décision prise le jour t s'arrête aux rendements de t-2.
# Le filtre de tendance, lui, voit la clôture de t-1.
COV_LAG = 2
TREND_LAG = 1


# ------------------------------------------------------------------ données

@dataclass
class Market:
    """Clôtures quotidiennes alignées (devise du portefeuille)."""

    days: list[date]
    symbols: list[str]
    closes: np.ndarray                      # (T, N)
    rates: np.ndarray                       # (T,) taux court annuel en vigueur ce jour
    interest: dict[str, np.ndarray] = field(default_factory=dict)   # intérêts cumulés par part (Reynders)

    def interest_payload(self) -> dict[str, list]:
        """Index d'intérêts au format de `TaxModel.set_interest_index` (jours de changement)."""
        out = {}
        for sym, cum in self.interest.items():
            rows = [[self.days[t].isoformat(), float(cum[t])] for t in range(len(self.days))
                    if cum[t] > 0 and (t == 0 or cum[t] != cum[t - 1])]
            if rows:
                out[sym] = rows
        return out


def load_market(dataset: str | Path, cfg: Config) -> Market:
    dataset = resolve_path(dataset)
    days, prices = daily_closes(read_closes(dataset))
    symbols = sorted(prices)
    closes = np.full((len(days), len(symbols)), np.nan)
    for j, sym in enumerate(symbols):
        last = np.nan
        for t, day in enumerate(days):
            last = prices[sym].get(day, last)
            closes[t, j] = last
    start = int(np.argmax(~np.isnan(closes).any(axis=1)))
    if np.isnan(closes[start]).any():
        raise ValueError(f"no day with a price for every symbol in {dataset}")
    days, closes = days[start:], closes[start:]

    rates = np.zeros(len(days))
    if cfg.financing.enabled:
        rp = rates_path(dataset)
        series = (RateSeries.from_payload(json.loads(rp.read_text(encoding="utf-8"))) if rp.exists()
                  else RateSeries.fixed(cfg.financing.fixed_rate, date(1970, 1, 1)))
        rates = np.array([series.rate(d) for d in days])

    interest: dict[str, np.ndarray] = {}
    meta = meta_path(dataset)
    if meta.exists():
        index = json.loads(meta.read_text(encoding="utf-8")).get("interest_index") or {}
        for sym, rows in index.items():
            if sym not in symbols:
                continue
            rows = sorted((date.fromisoformat(d), float(v)) for d, v in rows)
            cum, k, cur = np.zeros(len(days)), 0, 0.0
            for t, day in enumerate(days):
                while k < len(rows) and rows[k][0] <= day:
                    cur = rows[k][1]
                    k += 1
                cum[t] = cur
            interest[sym] = cum
    return Market(days, symbols, closes, rates, interest)


def make_tax_model(cfg: Config, market: Market) -> TaxModel | None:
    """Modèle fiscal neuf (un registre de lots par simulation), comme `runner._tax_model`."""
    if not cfg.tax.profile:
        return None
    profile = load_tax_profile(resolve_path(cfg.tax.profile))
    if cfg.tax.as_if_current_rules:
        profile = as_if_current_rules(profile)
    with_fx = cfg.fx.provider != "off"
    model = TaxModel(profile, cfg.instruments, portfolio_currency=cfg.fx.portfolio_currency if with_fx else None)
    if with_fx:
        if cfg.fx.portfolio_currency != "EUR":
            raise ValueError("fastsim: portefeuille en EUR seulement (dataset importé avec --currency EUR)")
        model.set_fx(FxRates("EUR", "EUR", {date(1970, 1, 1): 1.0}, source="portefeuille en EUR"))
    payload = market.interest_payload()
    if payload:
        model.set_interest_index(payload)
    return model


# ------------------------------------------------------------------ calculs en lot

def ewma_covariances(closes: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Covariance EWMA des rendements log, comme `features.correlations.EWMACovariance`.

    Retourne (cov (T, N, N) par séance, nombre de mises à jour (T,))."""
    T, n = closes.shape
    r = np.zeros((T, n))
    r[1:] = np.log(closes[1:] / closes[:-1])
    cov = np.zeros((T, n, n))
    updates = np.zeros(T, dtype=int)
    for t in range(1, T):
        outer = np.outer(r[t], r[t])
        cov[t] = outer if t == 1 else lam * cov[t - 1] + (1.0 - lam) * outer
        updates[t] = t
    return cov, updates


def condition_covariances(cov: np.ndarray, scale: float, shrinkage: float) -> np.ndarray:
    """Annualisation, shrinkage vers la diagonale, projection PSD (`risk.covariance`), en lot."""
    c = cov * scale
    diag = np.einsum("tii->ti", c)
    shrunk = (1.0 - shrinkage) * c
    idx = np.arange(c.shape[1])
    shrunk[:, idx, idx] += shrinkage * diag
    shrunk = 0.5 * (shrunk + np.swapaxes(shrunk, 1, 2))
    values, vectors = np.linalg.eigh(shrunk)
    values = np.maximum(values, 0.0)
    return (vectors * values[:, None, :]) @ np.swapaxes(vectors, 1, 2)


def risk_budget_weights_batch(covs: np.ndarray, budgets: np.ndarray, *, max_iter: int = 100,
                              tol: float = 1e-12) -> np.ndarray:
    """Poids de budgets de risque pour K covariances à la fois (Newton amorti).

    Même solution que `allocation.risk_parity.risk_budget_weights` : le minimum
    de ½ xᵀΣx − Σ bᵢ ln xᵢ, qui vérifie xᵢ (Σx)ᵢ = bᵢ, puis w = x / Σx. Le
    problème est convexe, la solution unique. Les matrices pour lesquelles
    Newton ne s'applique pas (budget ou variance nuls) ou ne converge pas
    passent par la fonction de référence."""
    covs = np.asarray(covs, dtype=float)
    K, n, _ = covs.shape
    b = np.asarray(budgets, dtype=float)
    b = b / b.sum()
    diag = np.einsum("kii->ki", covs)
    ok = (diag > 0).all(axis=1) & bool((b > 0).all())
    out = np.zeros((K, n))
    if ok.any():
        c = covs[ok]
        x = np.sqrt(b)[None, :] / np.sqrt(diag[ok])
        done = np.zeros(len(c), dtype=bool)
        eye = np.eye(n)
        for _ in range(max_iter):
            sx = np.einsum("kij,kj->ki", c, x)
            resid = x * sx - b
            done = np.abs(resid).max(axis=1) < tol
            if done.all():
                break
            grad = sx - b / x
            hess = c + eye[None] * (b / (x * x))[:, :, None]
            step = np.linalg.solve(hess, -grad[:, :, None])[:, :, 0]
            # Pas le plus long qui garde x > 0 (au plus 1).
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(step < 0, -x / step, np.inf).min(axis=1)
            alpha = np.minimum(1.0, 0.9 * room)
            alpha[done] = 0.0
            x = x + alpha[:, None] * step
        w = x / x.sum(axis=1, keepdims=True)
        failed = ~done
        if failed.any():
            for k in np.flatnonzero(failed):
                w[k] = risk_budget_weights(c[k], b)
        out[ok] = w
    for k in np.flatnonzero(~ok):
        if (diag[k] > 0).any():
            out[k] = risk_budget_weights(covs[k], b)
    return out


def robustness_scores(covs: np.ndarray, base: np.ndarray, budgets: np.ndarray, vol_bump: float) -> np.ndarray:
    """Score de `robustness.stress.stress_test` pour chaque séance, sans signaux."""
    K, n, _ = covs.shape
    std = np.sqrt(np.clip(np.einsum("kii->ki", covs), 0.0, None))
    outer = std[:, :, None] * std[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer > 0, covs / outer, 0.0)
    idx = np.arange(n)
    scenarios = []
    for towards in (1.0, 0.0):                           # corr_up, corr_down
        new = corr + 0.5 * (towards - corr)
        new[:, idx, idx] = 1.0
        scenarios.append(new * outer)
    for i in range(n):                                   # vol_up:<symbole>
        bumped = covs.copy()
        bumped[:, i, :] *= vol_bump
        bumped[:, :, i] *= vol_bump
        scenarios.append(bumped)
    stacked = np.stack(scenarios, axis=1).reshape(K * len(scenarios), n, n)
    shapes = risk_budget_weights_batch(stacked, budgets).reshape(K, len(scenarios), n)
    total = base.sum(axis=1, keepdims=True)
    base_shape = np.where(total > 0, base / np.where(total > 0, total, 1.0), 0.0)
    instability = np.abs(shapes - base_shape[:, None, :]).sum(axis=2)
    empty = shapes.sum(axis=2) <= 0
    instability[empty] = 2.0
    # Un actif de variance nulle n'a pas de scénario vol_up dans le moteur.
    no_var = np.einsum("kii->ki", covs) <= 0
    instability[:, 2:][no_var] = 0.0
    score = np.clip(1.0 - instability.max(axis=1) / 2.0, 0.0, 1.0)
    score[total[:, 0] <= 0] = 1.0
    return score


@dataclass
class Precomputed:
    """Tout ce qui ne dépend que des prix, indexé par la séance dont la clôture est connue."""

    cov: np.ndarray            # (T, N, N) annualisée, conditionnée
    valid: np.ndarray          # (T,) covariance utilisable
    requested: np.ndarray      # (T, N) poids après ciblage de volatilité et bande d'exposition
    robustness: np.ndarray     # (T,)
    trend: np.ndarray          # (T, N) multiplicateurs de tendance après la clôture t
    budgets: np.ndarray


def precompute(market: Market, cfg: Config) -> Precomputed:
    alloc = cfg.allocation
    if alloc.method not in ("risk_parity", "class_parity"):
        raise ValueError(f"fastsim: méthode d'allocation {alloc.method!r} non prise en charge")
    T, n = market.closes.shape
    raw, updates = ewma_covariances(market.closes, cfg.ewma_lambda)
    cov = condition_covariances(raw, periods_per_year(cfg.risk.timeframe), cfg.risk.shrinkage)
    valid = updates >= max(cfg.risk.min_observations, 1)
    valid &= np.einsum("tii->ti", cov).max(axis=1) > 0

    allocator = RiskAllocator(alloc.method, target_vol=alloc.target_vol, max_gross=alloc.constraints.max_gross,
                              classes=alloc.constraints.sectors, class_budgets=alloc.class_budgets)
    budgets = (allocator.class_budgets(market.symbols) if alloc.method == "class_parity"
               else np.full(n, 1.0 / n))
    base = np.zeros((T, n))
    base[valid] = risk_budget_weights_batch(cov[valid], budgets)
    vol = np.sqrt(np.clip(np.einsum("ti,tij,tj->t", base, cov, base), 0.0, None))
    gross = np.abs(base).sum(axis=1)
    scale = np.ones(T)
    live = (vol > 0) & (gross > 0)
    scale[live] = np.minimum(alloc.target_vol / vol[live], alloc.constraints.max_gross / gross[live])
    requested = base * scale[:, None]

    # Bande d'exposition : niveau gardé tant que le ciblage ne le déplace pas assez.
    held = None
    for t in np.flatnonzero(valid):
        g = float(np.abs(requested[t]).sum())
        if alloc.gross_band <= 0 or g <= 0:
            held = g or held
        elif held is not None and held > 0 and abs(g / held - 1.0) < alloc.gross_band:
            requested[t] *= held / g
        else:
            held = g

    robustness = np.ones(T)
    if cfg.robustness.enabled:
        robustness[valid] = robustness_scores(cov[valid], base[valid], budgets, cfg.robustness.vol_bump)

    trend = np.ones((T, n))
    if alloc.trend.enabled:
        tf = TrendFilter(alloc.trend.horizons, alloc.trend.floor, update_every=alloc.trend.update_every,
                         min_change=alloc.trend.min_change)
        for t in range(T):
            for j, sym in enumerate(market.symbols):
                tf.on_close(sym, float(market.closes[t, j]))
            # Le moteur lit le taux du jour de la décision (le lendemain de cette clôture).
            rate = float(market.rates[min(t + TREND_LAG, T - 1)])
            mult = tf.multipliers(market.symbols, rate)
            trend[t] = [mult[s] for s in market.symbols]
    return Precomputed(cov, valid, requested, robustness, trend, budgets)


# ------------------------------------------------------------------ simulation

def _drawdown_scale(alloc, drawdown: float, portfolio_vol: float | None) -> float:
    """`Engine._drawdown_scale` : Grossman-Zhou / CPPI avec marge de chute."""
    limit = alloc.drawdown_control
    if not limit:
        return 1.0
    dd = -min(0.0, drawdown)
    if alloc.drawdown_gap_sigmas > 0 and portfolio_vol:
        gap = alloc.drawdown_gap_sigmas * portfolio_vol * math.sqrt(alloc.drawdown_gap_days / 252)
        limit = max(0.4 * limit, limit - gap)
    if dd >= limit:
        scale = 0.0
    else:
        scale = alloc.drawdown_multiplier * (1.0 - (1.0 - limit) / (1.0 - dd)) / limit
    return max(alloc.drawdown_min_scale, min(1.0, scale))


def simulate_profile(market: Market, cfg: Config, *, name: str = "Profil (simulateur rapide)",
                     pre: Precomputed | None = None, exec_lag: int = 0) -> StrategyResult:
    """Rejoue le profil sur `market`. Résultat au format des références du backtest."""
    pre = pre or precompute(market, cfg)
    alloc, fin = cfg.allocation, cfg.financing
    symbols, T, n = market.symbols, len(market.days), len(market.symbols)
    constraints = ConstraintEngine(alloc.constraints)
    hard = HardControls(cfg.hard_controls)
    tax = make_tax_model(cfg, market)
    decider = DecisionEngine(cfg.decision, tax, ExecutionCostModel(cfg.execution.cost))
    cost_rate = (cfg.execution.cost.default_spread_bps / 2 + cfg.execution.cost.slippage_bps) * 1e-4
    tracker = (RollingDrawdown(alloc.drawdown_window_days) if alloc.drawdown_window_days
               else DecayingPeakDrawdown(alloc.drawdown_peak_decay) if alloc.drawdown_peak_decay
               else DrawdownTracker())
    max_gross = gross_cap_of(alloc.constraints)
    lim = cfg.hard_controls
    harvest_from = tuple(int(x) for x in cfg.tax.harvest_from.split("-")) if cfg.tax.harvest_exemption else None

    qty = np.zeros(n)
    cash = float(cfg.portfolio.cash)
    target: dict[str, float] | None = None
    robustness: float | None = None
    tax_year = harvested_year = None
    result = StrategyResult(name, [], kind="fastsim")
    stats = {"gains_tax_paid": 0.0, "interest_tax_paid": 0.0, "harvested_gains": 0.0, "financing_cost": 0.0,
             "interest_earned": 0.0, "decisions": 0, "rebalances": 0, "forced_reductions": 0,
             "rejected_targets": 0}
    gross_sum = gross_max = 0.0

    for t, day in enumerate(market.days):
        prices = market.closes[t]
        # Intérêts du cash et du levier depuis la séance précédente.
        if fin.enabled and t > 0:
            amount = accrue(cash, float(market.rates[t - 1]), (day - market.days[t - 1]).days,
                            borrow_spread=fin.borrow_spread, credit_spread=fin.credit_spread,
                            credit_cash=fin.credit_cash)
            cash += amount
            stats["interest_earned" if amount >= 0 else "financing_cost"] += abs(amount)
        # Changement d'année : taxe sur les plus-values de l'année écoulée.
        if tax is not None and tax.gains is not None:
            if tax_year is None:
                tax_year = day.year
            while tax_year < day.year:
                due = tax.gains_tax_due(tax_year)
                if due > 0 and cfg.tax.settle_gains_tax:
                    tax.gains_taxes_paid += due
                    amount = tax.to_portfolio(due, day)
                    cash -= amount
                    stats["gains_tax_paid"] += amount
                tax_year += 1

        value = cash + float(qty @ prices)
        if isinstance(tracker, DrawdownTracker):
            tracker.update(value, None)
        else:
            tracker.update(value, day)
        weights = qty * prices / value if value > 0 else np.zeros(n)

        j = t - COV_LAG
        if j >= 0 and pre.valid[j] and value > 0:
            cov = pre.cov[j]
            current = {s: float(weights[i]) for i, s in enumerate(symbols)}
            requested = pre.requested[j] * pre.trend[t - TREND_LAG]
            port_vol = math.sqrt(max(float(weights @ cov @ weights), 0.0))
            dd_scale = _drawdown_scale(alloc, tracker.drawdown, port_vol)
            if dd_scale < 1.0:
                requested = requested * dd_scale
            rob = float(pre.robustness[j]) if cfg.robustness.enabled else None
            if rob is not None and rob < cfg.robustness.min_score:
                requested = weights + rob * (requested - weights)
            constrained = constraints.apply({s: float(requested[i]) for i, s in enumerate(symbols)},
                                            current, cov, symbols)
            if hard.validate_targets(constrained.weights).approved:
                target, robustness = dict(constrained.weights), rob
            else:
                stats["rejected_targets"] += 1

            # Récolte de l'exonération annuelle (vente puis rachat immédiat).
            if harvest_from and tax is not None and tax.gains is not None and harvested_year != day.year \
                    and (day.month, day.day) >= harvest_from:
                harvested_year = day.year
                positions = {s: (float(qty[i]), float(prices[i])) for i, s in enumerate(symbols)}
                for sym, q, gain in tax.harvest_plan(positions, day, extra_cost_rate=cost_rate,
                                                     min_benefit=cfg.tax.harvest_min_benefit):
                    price = positions[sym][1]
                    for signed in (-q, q):
                        charge, _ = tax.record_fill(sym, signed, price, day)
                        cash -= (charge.portfolio_amount or 0.0) + charge.interest_tax + q * price * cost_rate
                        stats["interest_tax_paid"] += charge.interest_tax
                    stats["harvested_gains"] += gain

            ctx = DecisionContext(
                timestamp=datetime.combine(day, time(20, 0), UTC), portfolio_value=value, current=current,
                target=target, symbols=symbols, cov=cov,
                prices={s: float(prices[i]) for i, s in enumerate(symbols)},
                robustness=robustness, portfolio_risk=port_vol,
                daily_vols={s: math.sqrt(max(cov[i, i], 0.0) / 252) for i, s in enumerate(symbols) if cov[i, i] > 0},
                max_gross=max_gross, drawdown_scale=dd_scale if alloc.drawdown_control else None,
            )
            decision = decider.decide(ctx)
            stats["decisions"] += 1
            if decision.action == "UREBALANCE":
                stats["rebalances"] += 1
                if decision.reasons and decision.reasons[0].startswith("réduction de risque forcée"):
                    stats["forced_reductions"] += 1
                c = alloc.constraints
                cash_available = cash + max(0.0, c.max_gross - 1.0) * value - c.min_cash * value
                turnover_left = lim.max_daily_turnover
                order = sorted(decision.symbols.items(), key=lambda kv: (kv[1].notional > 0, kv[0]))
                fill_prices = market.closes[max(t - exec_lag, 0)]
                for sym, d in order:
                    i = symbols.index(sym)
                    price = float(fill_prices[i])
                    if abs(d.notional) < 1e-9 or price <= 0:
                        continue
                    q = float(math.trunc(d.notional / price))
                    # Plafonds des hard controls : taille d'ordre, part du portefeuille, turnover du jour.
                    cap = math.trunc(min(lim.max_order_notional, lim.max_order_weight * value,
                                         max(0.0, turnover_left) * value) / price)
                    q = max(-cap, min(cap, q))
                    if q < 0 and c.min_weight >= 0:
                        q = -min(-q, max(float(qty[i]), 0.0))
                    elif q > 0:
                        q = min(q, float(math.trunc(max(0.0, cash_available) / price)))
                    if q == 0:
                        continue
                    notional = q * price
                    cost = abs(notional) * cost_rate
                    if tax is not None:
                        charge, _ = tax.record_fill(sym, q, price, day)
                        cost += (charge.portfolio_amount or 0.0) + charge.interest_tax
                        stats["interest_tax_paid"] += charge.interest_tax
                        result.taxes.append((day, charge.amount))
                    qty[i] += q
                    cash -= notional + cost
                    if q > 0:
                        cash_available -= notional
                    turnover_left -= abs(notional) / value
                    result.trades.append((day, abs(notional)))
                value = cash + float(qty @ prices)

        gross = float(np.abs(qty * prices).sum()) / value if value > 0 else 0.0
        gross_sum += gross
        gross_max = max(gross_max, gross)
        result.equity.append((day, value))
        result.exposure.append((day, gross))

    stats["avg_gross"] = gross_sum / T if T else 0.0
    stats["max_gross"] = gross_max
    stats["fills"] = len(result.trades)
    if tax is not None and T:
        positions = {s: (float(qty[i]), float(market.closes[-1, i])) for i, s in enumerate(symbols) if qty[i] > 0}
        stats["liquidation_cost"] = (tax.liquidation_cost(positions, market.days[-1])
                                     + sum(q * p for q, p in positions.values()) * cost_rate)
    result.stats = stats
    return result


# ------------------------------------------------------------------ distribution par bootstrap

def block_bootstrap_market(market: Market, block: int, rng: np.random.Generator) -> Market:
    """Trajectoire simulée : blocs contigus de `block` séances de rendements
    tirés dans l'historique, conjointement pour les 8 actifs (on garde les
    corrélations et les épisodes de crise, mais dans un ordre différent).

    Les taux du cash suivent les mêmes blocs. L'index d'intérêts (Reynders) n'est
    pas rééchantillonné : le modèle fiscal replie alors sur la règle prudente
    (plus-value × part en créances), donc l'impôt simulé est un peu plus élevé."""
    T = len(market.days)
    if T < block + 2:
        raise ValueError("historique trop court pour ce bloc")
    R = np.diff(np.log(market.closes), axis=0)                 # (T-1, N) ; R[k] : séance k -> k+1
    n_blocks = -(-(T - 1) // block)
    starts = rng.integers(0, len(R) - block + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[: T - 1]
    closes = np.empty_like(market.closes)
    closes[0] = market.closes[0]
    closes[1:] = market.closes[0] * np.exp(np.cumsum(R[idx], axis=0))
    rates = np.empty_like(market.rates)
    rates[0] = market.rates[0]
    rates[1:] = market.rates[idx + 1]
    return Market(market.days, market.symbols, closes, rates, {})


def monte_carlo_profile(market: Market, cfg: Config, *, paths: int, block: int = 21, seed: int = 0) -> list[dict]:
    """Un résultat par trajectoire : rendement, rendement liquidé, perte maximale."""
    rng = np.random.default_rng(seed)
    years = (market.days[-1] - market.days[0]).days / 365.25
    start_cash = float(cfg.portfolio.cash)
    out = []
    for _ in range(paths):
        sim = block_bootstrap_market(market, block, rng)
        run = simulate_profile(sim, cfg, pre=precompute(sim, cfg))
        final = run.equity[-1][1]
        liq = run.stats.get("liquidation_cost") or 0.0
        final_liq = final - liq
        mm = metrics.compute(run.equity)
        out.append({
            "cagr": mm["cagr"], "cagr_liquide": (final_liq / start_cash) ** (1 / years) - 1 if final_liq > 0 else -1.0,
            "max_drawdown": mm["max_drawdown"], "sharpe": mm["sharpe"], "avg_gross": run.stats["avg_gross"],
            "fills": run.stats["fills"],
        })
    return out


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def summarize_paths(results: list[dict]) -> dict:
    def q(key: str) -> dict[str, float]:
        vals = np.array([r[key] for r in results], dtype=float)
        return {f"p{int(p * 100):02d}": float(np.quantile(vals, p)) for p in QUANTILES}

    dd = np.array([r["max_drawdown"] for r in results], dtype=float)
    liq = np.array([r["cagr_liquide"] for r in results], dtype=float)
    return {
        "paths": len(results),
        "cagr_liquide": q("cagr_liquide"),
        "max_drawdown": q("max_drawdown"),
        "sharpe": q("sharpe"),
        "p_max_drawdown_below_15pct": float((dd < -0.15).mean()),
        "p_max_drawdown_below_20pct": float((dd < -0.20).mean()),
        "p_cagr_liquide_negative": float((liq < 0).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    from trading_engine.backtest.runner import with_params
    from trading_engine.config import DEFAULT_CONFIG_PATH, load_config

    p = argparse.ArgumentParser(description="Distribution du profil par bootstrap par blocs")
    p.add_argument("--data", default="data/backtest/daily_eur.jsonl")
    p.add_argument("--paths", type=int, default=200)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/backtest/montecarlo.json")
    args = p.parse_args(argv)
    overlays = ("config/backtest_daily.yaml", "config/profil_belge.yaml")
    cfg = load_config(DEFAULT_CONFIG_PATH, tuple(resolve_path(o) for o in overlays))
    cfg = with_params(cfg, {"allocation.gross_band": 0.10})
    market = load_market(args.data, cfg)
    results = monte_carlo_profile(market, cfg, paths=args.paths, block=args.block, seed=args.seed)
    summary = summarize_paths(results)
    summary.update(block=args.block, seed=args.seed, days=len(market.days))
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "paths": results}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
