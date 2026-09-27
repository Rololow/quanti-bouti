"""Chargement de la configuration YAML en objets typés et immuables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from trading_engine.allocation.constraints import Constraints
from trading_engine.risk.limits import RiskLimits

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


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
    replay_path: str | None = None
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)


@dataclass(frozen=True)
class PositionConfig:
    quantity: float
    avg_price: float


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


ALLOCATION_METHODS = ("static", "baseline", "risk_parity", "hrp", "signal")


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
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    constraints: Constraints = field(default_factory=Constraints)

    def __post_init__(self) -> None:
        if self.method not in ALLOCATION_METHODS:
            raise ValueError(
                f"unknown allocation method {self.method!r}, expected one of {ALLOCATION_METHODS}"
            )


@dataclass(frozen=True)
class RiskConfig:
    timeframe: str = "1h"          # barres utilisées pour la covariance
    shrinkage: float = 0.1         # vers la diagonale
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


@dataclass(frozen=True)
class FactorConfig:
    enabled: bool = True
    timeframe: str = "5m"
    horizon_bars: int = 6
    features: tuple[str, ...] = ("mom_30m", "mom_1h", "z_5m", "reversal_5m", "vwap_dist")
    forgetting: float = 0.995
    ridge: float = 10.0
    min_samples: int = 50


@dataclass(frozen=True)
class ModelsConfig:
    regimes: RegimesConfig = field(default_factory=RegimesConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)


@dataclass(frozen=True)
class StorageConfig:
    event_log: str | None = None


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
        risk = dict(raw.get("risk") or {})
        risk_limits = RiskLimits(**(risk.pop("limits", None) or {}))
        storage = raw.get("storage") or {}
        models = raw.get("models") or {}
        factor = dict(models.get("factor") or {})
        if "features" in factor:
            factor["features"] = tuple(factor["features"])

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
                replay_path=feed.get("replay_path"),
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
                    sym: PositionConfig(float(p["quantity"]), float(p["avg_price"]))
                    for sym, p in (pf.get("positions") or {}).items()
                },
                target_weights={
                    sym: float(w) for sym, w in (pf.get("target_weights") or {}).items()
                },
            ),
            allocation=AllocationConfig(
                baseline=alloc_baseline, constraints=alloc_constraints, **alloc
            ),
            risk=RiskConfig(limits=risk_limits, **risk),
            storage=StorageConfig(event_log=storage.get("event_log")),
            models=ModelsConfig(
                regimes=RegimesConfig(**(models.get("regimes") or {})),
                factor=FactorConfig(**factor),
            ),
        )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as fh:
        return Config.from_dict(yaml.safe_load(fh) or {})
