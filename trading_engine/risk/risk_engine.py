"""Risk Engine (README §15-16) : état de risque global du portefeuille.

Indépendant des modèles de signal. À chaque évaluation :

- volatilité ex-ante du portefeuille et contributions au risque ;
- concentration (nombre effectif de positions, de paris en risque) ;
- diversification ratio ;
- drawdown du portefeuille et des positions ;
- levier ;
- dépassements de limites (`RiskBreach`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.portfolio.portfolio import PortfolioState
from trading_engine.risk.covariance import annualized_covariance
from trading_engine.risk.limits import RiskBreach, RiskLimits
from trading_engine.risk.portfolio_risk import (
    DrawdownTracker,
    diversification_ratio,
    effective_n,
    risk_contributions,
)
from trading_engine.risk.position_risk import PositionRisk


@dataclass(frozen=True)
class RiskReport:
    timestamp: datetime | None
    portfolio_vol: float | None
    drawdown: float
    max_drawdown: float
    leverage: float
    effective_positions: float        # 1 / HHI(poids)
    effective_bets: float | None      # 1 / HHI(parts de risque)
    diversification_ratio: float | None
    positions: dict[str, PositionRisk]
    breaches: tuple[RiskBreach, ...]


class RiskEngine:
    def __init__(
        self,
        features: FeatureEngine,
        *,
        timeframe: str = "1h",
        shrinkage: float = 0.1,
        limits: RiskLimits | None = None,
    ) -> None:
        self.features = features
        self.timeframe = timeframe
        self.shrinkage = shrinkage
        self.limits = limits or RiskLimits()
        self.portfolio_drawdown = DrawdownTracker()
        self._price_drawdowns: dict[str, DrawdownTracker] = {}

    def on_value(self, total_value: float, timestamp: datetime | None = None) -> None:
        self.portfolio_drawdown.update(total_value, timestamp)

    def on_price(self, symbol: str, price: float, timestamp: datetime | None = None) -> None:
        self._price_drawdowns.setdefault(symbol, DrawdownTracker()).update(price, timestamp)

    def covariance(self, symbols: list[str]) -> np.ndarray | None:
        return annualized_covariance(self.features, symbols, self.timeframe, shrinkage=self.shrinkage)

    def evaluate(self, state: PortfolioState) -> RiskReport:
        total = state.total_value
        symbols = [p.symbol for p in state.positions]
        weights = np.array([p.weight for p in state.positions])
        cov = self.covariance(symbols) if symbols else None

        decomposition = risk_contributions(weights, cov) if cov is not None else None
        vols = np.sqrt(np.clip(np.diag(cov), 0.0, None)) if cov is not None else None

        positions: dict[str, PositionRisk] = {}
        for i, p in enumerate(state.positions):
            dd = self._price_drawdowns.get(p.symbol)
            positions[p.symbol] = PositionRisk(
                symbol=p.symbol,
                weight=p.weight,
                target_weight=p.target_weight,
                volatility=None if vols is None or vols[i] == 0 else float(vols[i]),
                marginal_risk=None if decomposition is None else float(decomposition.marginal[i]),
                risk_contribution=None if decomposition is None else float(decomposition.contribution[i]),
                risk_share=None if decomposition is None else float(decomposition.relative[i]),
                drawdown=0.0 if dd is None else dd.drawdown,
            )

        port_vol = None if decomposition is None else decomposition.portfolio_vol
        effective_bets = (
            effective_n(decomposition.relative)
            if decomposition is not None and port_vol and port_vol > 0 else None
        )
        report = RiskReport(
            timestamp=state.timestamp,
            portfolio_vol=port_vol,
            drawdown=self.portfolio_drawdown.drawdown,
            max_drawdown=self.portfolio_drawdown.max_drawdown,
            leverage=state.gross_exposure / total if total else 0.0,
            effective_positions=effective_n(weights) if weights.size else 0.0,
            effective_bets=effective_bets,
            diversification_ratio=None if cov is None else diversification_ratio(weights, cov),
            positions=positions,
            breaches=(),
        )
        return RiskReport(**{**vars(report), "breaches": tuple(self.check(report))})

    def check(self, report: RiskReport) -> list[RiskBreach]:
        lim = self.limits
        breaches: list[RiskBreach] = []
        if report.portfolio_vol is not None and report.portfolio_vol > lim.max_portfolio_vol:
            breaches.append(RiskBreach("PORTFOLIO_VOL", None, report.portfolio_vol, lim.max_portfolio_vol))
        if -report.drawdown > lim.max_drawdown:
            breaches.append(RiskBreach("DRAWDOWN", None, -report.drawdown, lim.max_drawdown))
        if report.leverage > lim.max_gross_leverage:
            breaches.append(RiskBreach("LEVERAGE", None, report.leverage, lim.max_gross_leverage))
        for sym, pos in sorted(report.positions.items()):
            if abs(pos.weight) > lim.max_weight:
                breaches.append(RiskBreach("WEIGHT", sym, abs(pos.weight), lim.max_weight))
            if pos.risk_share is not None and pos.risk_share > lim.max_risk_contribution:
                breaches.append(RiskBreach("RISK_CONTRIBUTION", sym, pos.risk_share, lim.max_risk_contribution))
            if pos.weight != 0 and -pos.drawdown > lim.max_position_drawdown:
                breaches.append(RiskBreach("POSITION_DRAWDOWN", sym, -pos.drawdown, lim.max_position_drawdown))
            if abs(pos.drift) > lim.max_drift:
                breaches.append(RiskBreach("DRIFT", sym, abs(pos.drift), lim.max_drift))
        return breaches
