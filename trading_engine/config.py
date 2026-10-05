"""Chargement de la configuration YAML en objets typés et immuables."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from trading_engine.alerts.alerts import AlertConfig
from trading_engine.allocation.constraints import Constraints
from trading_engine.decision.rebalance import DecisionConfig
from trading_engine.data.integrity import IntegrityConfig
from trading_engine.execution.cost_model import CostConfig
from trading_engine.execution.hard_controls import HardLimits
from trading_engine.execution.optimizer import OptimizerConfig
from trading_engine.safety.safety_engine import SafetyConfig
from trading_engine.tax.profile import Instrument
from trading_engine.risk.limits import RiskLimits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def resolve_path(path: str | Path) -> Path:
    """Chemin absolu, ou relatif au répertoire courant, sinon à la racine du projet."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


@dataclass(frozen=True)
class EngineConfig:
    max_events: int | None = None
    report_every: int = 250


@dataclass(frozen=True)
class AlpacaConfig:
    data_feed: str = "iex"
    url: str | None = None
    trades: bool = True
    quotes: bool = False
    bars: bool = True
    heartbeat_interval: float = 20.0
    heartbeat_timeout: float = 10.0
    backoff_initial: float = 1.0
    backoff_max: float = 60.0


@dataclass(frozen=True)
class FeedConfig:
    provider: str = "simulated"
    symbols: tuple[str, ...] = ()
    seed: int | None = None
    tick_seconds: float = 1.0
    annual_vol: float = 0.20
    sim_speed: float = 0.0             # 0 = max ; 1 = temps réel ; 60 = 60× plus vite
    replay_path: str | None = None
    dataset_path: str | None = None    # provider dataset : barres historiques (backtest)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)


@dataclass(frozen=True)
class PositionConfig:
    quantity: float
    avg_price: float
    acquired: date | None = None       # date d'achat (lots fiscaux) ; None = inconnue


@dataclass(frozen=True)
class PortfolioConfig:
    cash: float = 0.0
    positions: Mapping[str, PositionConfig] = field(default_factory=dict)
    target_weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FeaturesConfig:
    momentum_horizons: tuple[str, ...] = ("5m", "30m", "1h", "1d", "20d", "60d", "252d")
    mean_reversion_timeframes: tuple[str, ...] = ("5m",)
    zscore_window: int = 20
    correlation_timeframes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BaselineConfig:
    momentum_horizon: str = "1h"
    vol_timeframe: str = "1h"
    target_vol: float = 0.10
    budget: float = 1.0
    max_weight: float = 0.40


@dataclass(frozen=True)
class TrendConfig:
    """Filtre de tendance (allocation/trend.py) : poids × part des horizons
    haussiers. `redistribute` : le risque retiré est redonné aux actifs en
    tendance (jusqu'à la vol cible et au plafond brut) au lieu d'aller en cash."""
    enabled: bool = False
    timeframe: str = "1d"
    horizons: tuple[int, ...] = (21, 63, 126, 252)
    floor: float = 0.0
    redistribute: bool = False
    update_every: int = 1          # clôtures entre deux mises à jour du score (5 = hebdo)
    min_change: float = 0.0        # écart minimal de score pour changer (0.5 = 2 horizons sur 4)

    def __post_init__(self) -> None:
        if not self.horizons or any(int(h) < 1 for h in self.horizons):
            raise ValueError("allocation.trend.horizons must be >= 1")
        if not 0 <= self.floor <= 1:
            raise ValueError("allocation.trend.floor must be in [0, 1]")
        if self.update_every < 1 or not 0 <= self.min_change <= 1:
            raise ValueError("allocation.trend.update_every must be >= 1, min_change in [0, 1]")


ALLOCATION_METHODS = ("static", "baseline", "risk_parity", "hrp", "signal", "class_parity")


@dataclass(frozen=True)
class AllocationConfig:
    # "static"      : poids cibles de portfolio.target_weights ;
    # "baseline"    : momentum + volatility targeting (référence d'ablation) ;
    # "risk_parity" / "hrp" : allocation par le risque seul ;
    # "signal"      : budgets de risque issus des signaux + risk parity.
    method: str = "static"
    rebalance_timeframe: str = "1h"
    target_vol: float = 0.10
    min_skill: float = 0.0
    # class_parity : part du risque par classe (constraints.sectors), ex.
    # {equity: 0.5, bonds: 0.25, commodities: 0.25} ; vide = parts égales.
    class_budgets: Mapping[str, float] = field(default_factory=dict)
    # Contrôle du drawdown (Grossman-Zhou) : l'exposition est réduite en
    # proportion de la marge restante avant ce drawdown (ex. 0.15) ; None = aucun.
    drawdown_control: float | None = None
    drawdown_min_scale: float = 0.0
    # Plus haut de référence du contrôle du drawdown : jours calendaires
    # glissants (ex. 365) ; None = plus haut historique (peut bloquer en cash).
    drawdown_window_days: int | None = None
    # Multiplicateur du coussin (CPPI) : 1 = Grossman-Zhou (l'exposition baisse
    # dès la première perte) ; 2 = pleine exposition jusqu'à la moitié de la
    # limite environ, puis réduction plus rapide jusqu'à 0 à la limite.
    drawdown_multiplier: float = 1.0
    trend: TrendConfig = field(default_factory=TrendConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    constraints: Constraints = field(default_factory=Constraints)

    def __post_init__(self) -> None:
        if self.method not in ALLOCATION_METHODS:
            raise ValueError(
                f"unknown allocation method {self.method!r}, expected one of {ALLOCATION_METHODS}"
            )
        if any(v < 0 for v in self.class_budgets.values()):
            raise ValueError("allocation.class_budgets must be >= 0")
        if self.drawdown_control is not None and not 0 < self.drawdown_control < 1:
            raise ValueError("allocation.drawdown_control must be in (0, 1)")
        if not 0 <= self.drawdown_min_scale <= 1:
            raise ValueError("allocation.drawdown_min_scale must be in [0, 1]")
        if self.drawdown_multiplier < 1:
            raise ValueError("allocation.drawdown_multiplier must be >= 1")
        if self.drawdown_window_days is not None and self.drawdown_window_days < 1:
            raise ValueError("allocation.drawdown_window_days must be >= 1")


@dataclass(frozen=True)
class RiskConfig:
    timeframe: str = "1h"          # barres utilisées pour la covariance
    shrinkage: float = 0.1         # vers la diagonale
    min_observations: int = 20     # barres minimum avant toute covariance / allocation
    limits: RiskLimits = field(default_factory=RiskLimits)


@dataclass(frozen=True)
class RegimesConfig:
    enabled: bool = True
    n_states: int = 3
    # HMM-HF / HMM-MT / HMM-LT -> timeframe de barre
    timeframes: Mapping[str, str] = field(
        default_factory=lambda: {"HF": "5m", "MT": "1h", "LT": "1d"}
    )
    min_samples: int = 100
    window: int = 500
    refit_every: int = 50
    degraded_z: float = 4.0        # MODEL_DEGRADED si la vraisemblance récente chute de z écarts-types


@dataclass(frozen=True)
class FactorConfig:
    name: str = "factor"
    enabled: bool = True
    timeframe: str = "5m"
    horizon_bars: int = 6
    features: tuple[str, ...] = ("mom_30m", "mom_1h", "z_5m", "reversal_5m", "vwap_dist")
    forgetting: float = 0.995
    ridge: float = 10.0
    min_samples: int = 50


@dataclass(frozen=True)
class EnsembleConfig:
    lam: float = 0.99              # oubli de la covariance des erreurs
    shrinkage: float = 0.5         # vers l'équipondération
    min_samples: int = 30          # avant : erreurs supposées parfaitement corrélées


@dataclass(frozen=True)
class ModelsConfig:
    regimes: RegimesConfig = field(default_factory=RegimesConfig)
    predictors: tuple[FactorConfig, ...] = (FactorConfig(),)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)


@dataclass(frozen=True)
class StorageConfig:
    event_log: str | None = None
    decision_log: str | None = None


@dataclass(frozen=True)
class RobustnessConfig:
    enabled: bool = True
    min_score: float = 0.7        # en dessous : cible jugée incertaine, mouvement réduit
    vol_bump: float = 1.5


@dataclass(frozen=True)
class ExecutionConfig:
    # off : pas d'ordres ; proposals : ordres proposés (pour un humain) ;
    # paper : exécution simulée localement sur le flux ;
    # alpaca_paper : ordres envoyés au compte PAPER Alpaca (fills réels simulés
    # par Alpaca, sans argent réel). Aucun compte réel n'est jamais utilisé.
    mode: str = "paper"
    volume_timeframe: str = "5m"
    max_participation: float = 0.10
    fill_share: float = 1.0
    learn_impact: bool = False
    cost: CostConfig = field(default_factory=CostConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # alpaca_paper
    sync_portfolio: bool = True        # cash et positions lus sur le compte au démarrage
    reconcile: bool = True             # rapprochement à chaque intervalle (sans ordre en cours)
    time_in_force: str = "day"

    def __post_init__(self) -> None:
        if self.mode not in ("off", "proposals", "paper", "alpaca_paper"):
            raise ValueError(
                f"execution.mode must be off, proposals, paper or alpaca_paper, got {self.mode!r}")


@dataclass(frozen=True)
class FundamentalsSourceConfig:
    # auto : simulé si le marché est simulé, sinon aucun (jamais de fausses
    # données mélangées à un flux réel) ; none | simulated | file | edgar
    provider: str = "auto"
    path: str | None = None
    weights: Mapping[str, float] | None = None
    half_life_days: float = 30.0
    edgar_user_agent: str = ""               # "Nom email@exemple.com" (exigé par la SEC)
    edgar_ciks: Mapping[str, int] = field(default_factory=dict)
    edgar_poll_every: float = 3600.0


@dataclass(frozen=True)
class NewsSourceConfig:
    provider: str = "auto"                   # auto | none | simulated | file | alpaca
    path: str | None = None
    similarity: float = 0.5
    window: str = "24h"
    activity_half_life: str = "6h"


@dataclass(frozen=True)
class QualitativeConfig:
    fundamentals: FundamentalsSourceConfig = field(default_factory=FundamentalsSourceConfig)
    news: NewsSourceConfig = field(default_factory=NewsSourceConfig)
    sim_earnings_every: str = "24h"
    sim_news_every: str = "3h"
    sim_duplicate_probability: float = 0.5


@dataclass(frozen=True)
class AISettings:
    # auto : simulé si le marché est simulé, sinon aucun (Claude coûte de
    # l'argent : il doit être activé explicitement) ; none | simulated | claude
    provider: str = "auto"
    model: str = "claude-opus-5"
    fast_effort: str = "low"
    escalated_effort: str = "high"
    escalate_below_confidence: float = 0.6
    min_confidence: float = 0.5
    max_calls_per_hour: int = 120
    important_threshold: float = 0.7
    extract_fundamentals: bool = True
    server_fallbacks: bool = True
    simulated_latency_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.provider not in ("auto", "none", "simulated", "claude"):
            raise ValueError(f"ai.provider must be auto, none, simulated or claude, got {self.provider!r}")


@dataclass(frozen=True)
class FinancingConfig:
    """Intérêts sur le cash et coût du levier (voir portfolio/financing.py).
    Désactivé par défaut ; en backtest, la série T-bill (^IRX) du dataset est
    utilisée si présente, sinon `fixed_rate`. Jamais appliqué avec un compte
    broker réel (le broker fait foi)."""

    enabled: bool = False
    fixed_rate: float = 0.0           # taux court annuel si aucune série
    borrow_spread: float = 0.005      # emprunt : taux court + 0,5 % (≈ financement implicite des futures)
    credit_cash: bool = True          # cash positif rémunéré
    credit_spread: float = 0.0        # cash : taux court - écart


@dataclass(frozen=True)
class CalendarConfig:
    """Séances de marché. `auto` : actif avec le flux Alpaca (calendrier
    Alpaca, règles NYSE en repli), inactif en simulation ; en replay, le
    calendrier vient du journal. `on` : règles NYSE hors Alpaca. `off` : 24/7."""

    mode: str = "auto"
    avoid_open_minutes: float = 5.0      # pas de décision juste après l'ouverture
    avoid_close_minutes: float = 15.0    # ni juste avant la clôture
    horizon_days: int = 30               # période chargée (renouvelée en live)

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "on", "off"):
            raise ValueError(f"calendar.mode must be auto|on|off, got {self.mode!r}")
        if self.avoid_open_minutes < 0 or self.avoid_close_minutes < 0 or self.horizon_days < 1:
            raise ValueError("calendar: minutes must be >= 0 and horizon_days >= 1")


@dataclass(frozen=True)
class WarmupConfig:
    """Historique chargé au démarrage (flux Alpaca) : jours calendaires par timeframe."""

    enabled: bool = True
    lookback_days: Mapping[str, float] = field(
        default_factory=lambda: {"5m": 5, "1h": 45, "1d": 400})


@dataclass(frozen=True)
class TaxConfig:
    profile: str | None = None                     # ex. config/taxes/BE.toml ; None = pas de fiscalité
    step_up_prices: Mapping[str, float] = field(default_factory=dict)   # devise du portefeuille
    # Registre fiscal persistant (lots datés, plus-values, TOB). Écrit seulement
    # en execution.mode alpaca_paper (jamais en simulation ni en replay).
    ledger_path: str | None = None
    # Backtest : règles d'aujourd'hui (taxe sur les plus-values) appliquées à
    # tout l'historique, sans step-up, pour mesurer leur effet.
    as_if_current_rules: bool = False
    # Taxe annuelle sur les plus-values prélevée sur le cash au changement
    # d'année (simulation) ; en live, comptée à part (à déclarer).
    settle_gains_tax: bool = True
    # Récolte de l'exonération annuelle (vente + rachat en décembre) : remonte
    # la base fiscale de gains exonérés. Simulation : exécutée ; live : alerte.
    harvest_exemption: bool = False
    harvest_from: str = "12-10"                   # MM-JJ : première séance de récolte
    harvest_min_benefit: float = 2.0              # impôt évité >= 2 × coût aller-retour


@dataclass(frozen=True)
class FxConfig:
    """Change devise du portefeuille -> devise du profil fiscal.
    auto : taux BCE avec le flux Alpaca, taux fixe en simulation ; en replay,
    les taux viennent du journal. ecb | fixed | off (pas de conversion)."""

    provider: str = "auto"
    portfolio_currency: str = "USD"
    fixed_rate: float = 1.10          # unités de devise du portefeuille pour 1 unité fiscale (USD par EUR)
    history_days: int = 400           # historique BCE chargé au démarrage

    def __post_init__(self) -> None:
        if self.provider not in ("auto", "ecb", "fixed", "off"):
            raise ValueError(f"fx.provider must be auto|ecb|fixed|off, got {self.provider!r}")
        if not self.fixed_rate > 0 or self.history_days < 1:
            raise ValueError("fx.fixed_rate must be > 0 and history_days >= 1")


@dataclass(frozen=True)
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    bar_timeframes: tuple[str, ...] = ("5m", "1h", "1d")
    ewma_lambda: float = 0.94
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    hard_controls: HardLimits = field(default_factory=HardLimits)
    tax: TaxConfig = field(default_factory=TaxConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    qualitative: QualitativeConfig = field(default_factory=QualitativeConfig)
    ai: AISettings = field(default_factory=AISettings)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    fx: FxConfig = field(default_factory=FxConfig)
    financing: FinancingConfig = field(default_factory=FinancingConfig)
    instruments: Mapping[str, Instrument] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        engine = raw.get("engine") or {}
        feed = raw.get("feed") or {}
        bars = raw.get("bars") or {}
        vol = raw.get("volatility") or {}
        pf = raw.get("portfolio") or {}
        feats = raw.get("features") or {}
        default_feats = FeaturesConfig()
        corr_tfs = feats.get("correlation_timeframes")
        alloc = dict(raw.get("allocation") or {})
        alloc_baseline = BaselineConfig(**(alloc.pop("baseline", None) or {}))
        alloc_constraints = Constraints(**(alloc.pop("constraints", None) or {}))
        trend = dict(alloc.pop("trend", None) or {})
        if "horizons" in trend:
            trend["horizons"] = tuple(int(h) for h in trend["horizons"])
        alloc_trend = TrendConfig(**trend)
        risk = dict(raw.get("risk") or {})
        risk_limits = RiskLimits(**(risk.pop("limits", None) or {}))
        safety = dict(raw.get("safety") or {})
        if "halt_on_breaches" in safety:
            safety["halt_on_breaches"] = tuple(safety["halt_on_breaches"])
        storage = raw.get("storage") or {}
        models = raw.get("models") or {}
        # "predictors" (liste) ; "factor" (un seul prédicteur) reste accepté.
        predictor_specs = models.get("predictors")
        if predictor_specs is None:
            predictor_specs = [models["factor"]] if models.get("factor") else [{}]
        predictors = []
        for spec in predictor_specs:
            spec = dict(spec)
            if "features" in spec:
                spec["features"] = tuple(spec["features"])
            predictors.append(FactorConfig(**spec))

        ewma_lambda = float(vol.get("lambda", 0.94))
        if not 0.0 < ewma_lambda < 1.0:
            raise ValueError(f"volatility.lambda must be in (0, 1), got {ewma_lambda}")

        return cls(
            engine=EngineConfig(
                max_events=engine.get("max_events"),
                report_every=int(engine.get("report_every", 250)),
            ),
            feed=FeedConfig(
                provider=feed.get("provider", "simulated"),
                symbols=tuple(feed.get("symbols", ())),
                seed=feed.get("seed"),
                tick_seconds=float(feed.get("tick_seconds", 1.0)),
                annual_vol=float(feed.get("annual_vol", 0.20)),
                sim_speed=float(feed.get("sim_speed", 0.0)),
                replay_path=feed.get("replay_path"),
                dataset_path=feed.get("dataset_path"),
                alpaca=AlpacaConfig(**(feed.get("alpaca") or {})),
            ),
            bar_timeframes=tuple(bars.get("timeframes", ("5m", "1h", "1d"))),
            ewma_lambda=ewma_lambda,
            features=FeaturesConfig(
                momentum_horizons=tuple(
                    feats.get("momentum_horizons", default_feats.momentum_horizons)
                ),
                mean_reversion_timeframes=tuple(
                    feats.get("mean_reversion_timeframes", default_feats.mean_reversion_timeframes)
                ),
                zscore_window=int(feats.get("zscore_window", default_feats.zscore_window)),
                correlation_timeframes=None if corr_tfs is None else tuple(corr_tfs),
            ),
            portfolio=PortfolioConfig(
                cash=float(pf.get("cash", 0.0)),
                positions={
                    sym: PositionConfig(float(p["quantity"]), float(p["avg_price"]), p.get("acquired"))
                    for sym, p in (pf.get("positions") or {}).items()
                },
                target_weights={
                    sym: float(w) for sym, w in (pf.get("target_weights") or {}).items()
                },
            ),
            allocation=AllocationConfig(
                baseline=alloc_baseline, constraints=alloc_constraints, trend=alloc_trend, **alloc
            ),
            risk=RiskConfig(limits=risk_limits, **risk),
            integrity=IntegrityConfig(**(raw.get("integrity") or {})),
            safety=SafetyConfig(**safety),
            hard_controls=HardLimits(**(raw.get("hard_controls") or {})),
            tax=TaxConfig(**(raw.get("tax") or {})),
            robustness=RobustnessConfig(**(raw.get("robustness") or {})),
            decision=DecisionConfig(**{
                k: tuple(v) if k == "fractions" else v
                for k, v in (raw.get("decision") or {}).items()
            }),
            execution=_execution_config(raw.get("execution") or {}),
            qualitative=_qualitative_config(raw.get("qualitative") or {}),
            ai=AISettings(**(raw.get("ai") or {})),
            warmup=WarmupConfig(**(raw.get("warmup") or {})),
            calendar=CalendarConfig(**(raw.get("calendar") or {})),
            fx=FxConfig(**(raw.get("fx") or {})),
            financing=FinancingConfig(**(raw.get("financing") or {})),
            alerts=AlertConfig(**{
                k: tuple(v) if k == "regime_horizons" else v
                for k, v in (raw.get("alerts") or {}).items()
            }),
            instruments={
                sym: Instrument(sym, **spec) for sym, spec in (raw.get("instruments") or {}).items()
            },
            storage=StorageConfig(event_log=storage.get("event_log"),
                                  decision_log=storage.get("decision_log")),
            models=ModelsConfig(
                regimes=RegimesConfig(**(models.get("regimes") or {})),
                predictors=tuple(predictors),
                ensemble=EnsembleConfig(**(models.get("ensemble") or {})),
            ),
        )


def _qualitative_config(raw: Mapping[str, Any]) -> QualitativeConfig:
    raw = dict(raw)
    fundamentals = FundamentalsSourceConfig(**(raw.pop("fundamentals", None) or {}))
    news = NewsSourceConfig(**(raw.pop("news", None) or {}))
    sim = raw.pop("simulated", None) or {}
    return QualitativeConfig(
        fundamentals=fundamentals, news=news,
        sim_earnings_every=sim.get("earnings_every", "24h"),
        sim_news_every=sim.get("news_every", "3h"),
        sim_duplicate_probability=float(sim.get("duplicate_probability", 0.5)),
    )


def _execution_config(raw: Mapping[str, Any]) -> ExecutionConfig:
    raw = dict(raw)
    cost = CostConfig(**(raw.pop("cost", None) or {}))
    opt = dict(raw.pop("optimizer", None) or {})
    for key in ("aggressiveness", "durations"):
        if key in opt:
            opt[key] = tuple(opt[key])
    return ExecutionConfig(cost=cost, optimizer=OptimizerConfig(**opt), **raw)


def deep_merge(base: dict, overlay: dict) -> dict:
    """Fusion récursive : les dictionnaires sont fusionnés, le reste (listes
    comprises) est remplacé par la valeur de la surcouche. Un dictionnaire
    vide remplace (ex. `positions: {}` = aucune position)."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and value and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, overlays: tuple[str | Path, ...] = ()) -> Config:
    """Config de base + surcouches éventuelles (ex. config/backtest.yaml)."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for overlay in overlays:
        with open(overlay, encoding="utf-8") as fh:
            raw = deep_merge(raw, yaml.safe_load(fh) or {})
    return Config.from_dict(raw)
