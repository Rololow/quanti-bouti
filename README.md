# quanti-bouti

# Real-Time Adaptive Quantitative Portfolio Engine

> **Real-time, multi-horizon, adaptive portfolio monitoring and decision engine combining quantitative market signals, online learning, regime detection, fundamentals, news and structured AI extraction.**

---

## Démarrage rapide

```bash
pip install -e ".[dev]"

# Lance la boucle Core sur un flux simulé (config/config.yaml)
python -m trading_engine.main

# Tests
pytest
```

État actuel : Phase 1 (Core), Phase 2 (Realtime), Phase 3 (Features :
rendements, volatilité EWMA, momentum multi-horizon normalisé, mean reversion,
VWAP, corrélations EWMA), Phase 4 (journal d'événements, replay déterministe,
baseline momentum + volatility targeting) et Phase 5 (HMM de régime HF / MT / LT,
modèle de facteurs online émettant des signaux « rendement attendu ± incertitude »,
avec oubli, régularisation, taille minimale d'échantillon, suivi du skill et
détection de dérive), Phase 6 (volatilité ex-ante, contributions au risque,
concentration, drawdown, limites et alertes) et Phase 7 (risk parity, HRP,
budgets de risque issus des signaux, volatility targeting, Constraint Engine,
attribution de la cible, drift monitoring), Phase 8 (Data Integrity avec
quarantaine des sauts non confirmés, Safety Engine NORMAL / DEGRADED / HALTED,
hard controls indépendants des modèles) et profils fiscaux par pays (TOML,
Belgique fournie).

Par défaut le moteur tourne sur un flux simulé déterministe. Pour le flux
Alpaca temps réel :

```bash
export APCA_API_KEY_ID=...        # voir .env.example
export APCA_API_SECRET_KEY=...
# puis dans config/config.yaml : feed.provider: alpaca, engine.max_events: null
python -m trading_engine.main
```

Enregistrer puis rejouer une session (même moteur, même résultat) :

```yaml
# config/config.yaml
storage:
  event_log: data/events.jsonl     # enregistre les événements bruts reçus
feed:
  provider: replay                 # puis rejoue le journal
  replay_path: data/events.jsonl
allocation:
  method: signal                   # static | baseline | risk_parity | hrp | signal
```

Le plan Alpaca gratuit donne accès au flux `iex`. Le client gère la
reconnexion (backoff exponentiel), le heartbeat (ping WebSocket en cas
d'inactivité) et horodate chaque événement (heure bourse + heure de réception).

---

## 1. Vision

L'objectif du projet est de construire un **moteur quantitatif de monitoring et de décision de portefeuille en temps réel**.

Le système suit en permanence :

* chaque position ;
* son exposition ;
* son PnL ;
* sa volatilité ;
* sa contribution au risque ;
* son momentum ;
* ses signaux multi-horizon ;
* le régime de marché ;
* les données fondamentales ;
* les événements/news ;
* les changements de corrélation ;
* l'écart entre allocation actuelle et allocation cible.

Le système doit pouvoir répondre à tout moment à :

> **"Quelle est la situation actuelle de chaque position, quel risque représente-t-elle, pourquoi sa cible devrait-elle changer et quels événements ont provoqué cette évolution ?"**

Le système est conçu autour de **l'apprentissage online**.

Le backtesting n'est donc **pas le moteur principal** du système. Il reste un outil de validation et d'analyse historique.

---

# 2. Philosophie générale

Architecture principale :

```text
                          REAL-TIME DATA
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    Market Data            Fundamentals               News
         └──────────────────────┼──────────────────────┘
                                ▼
                         DATA INTEGRITY
                  (sanity checks, quarantaine,
                   score de qualité par source)
                                │
                                ▼
                         FEATURE ENGINE
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    HMM REGIMES          PREDICTIVE MODELS         NLP / LLM
  (conditionnement)    (factor, TSFM optionnel)   (événements)
         └──────────────────────┼──────────────────────┘
                                ▼
                         MODEL ENSEMBLE
                     ┌──────────┴──────────┐
                     ▼                     ▼
                PREDICTION            RELIABILITY
                  μ ± σ           (skill par contexte)
                     └──────────┬──────────┘
                                ▼
                          SIGNAL FUSION
                  (corrélation des erreurs entre
                   modèles, pas de double comptage)
                                │
                                ▼
                       ROBUSTNESS ENGINE
              (stress tests, désaccord, dégradation)
                                │
                                ▼
                          RISK ENGINE
                                │
                                ▼
                       ALLOCATION ENGINE
                                │
                                ▼
                       CONSTRAINT ENGINE
                                │
                                ▼
                    UREBALANCE / UDONOTHING
                          │           │
                          ▼           └──► aucune action
                    EXECUTION ENGINE
                          │
                          ▼
                     HARD CONTROLS
            (limites indépendantes des modèles)
                          │
                          ▼
                    ORDER PROPOSAL
                          │
                          ▼
                Human / Broker Execution
                          │
                          ▼
                  EXECUTION OBSERVED
                ┌─────────┴─────────┐
                ▼                   ▼
          ALPHA LEARNING     EXECUTION LEARNING


  SAFETY ENGINE : NORMAL / DEGRADED / HALTED
  supervise toute la chaîne (données, modèles, risque, ordres) ;
  en HALTED, seul UDONOTHING est possible.
```

Ce n'est pas un bot qui cherche des occasions de `BUY` / `SELL`. C'est un
**système de contrôle adaptatif du portefeuille** : il maintient en permanence
un état cible, mesure le coût de s'en écarter, décide s'il faut le corriger,
puis optimise la manière de réaliser cette correction.

---

# 3. Principe fondamental : monitoring ≠ trading automatique

Le système ne doit pas directement transformer un signal en ordre.

Il n'y a **aucun `BUY` / `SELL` / `HOLD` au niveau décisionnel**. Les seules
décisions possibles sont :

```text
UREBALANCE     revenir (partiellement) vers la cible
UDONOTHING     accepter la situation actuelle malgré le drift
```

Pipeline :

```text
SIGNAL
   ↓
TARGET WEIGHT
   ↓
RISK CHECK
   ↓
CONSTRAINTS
   ↓
UREBALANCE / UDONOTHING
   ↓
EXECUTION ENGINE      (uniquement après UREBALANCE)
   ↓
ORDER PROPOSAL
   ↓
HUMAN / BROKER EXECUTION
```

Un `ORDER_PROPOSAL` est une recommandation technique du moteur, pas
nécessairement un ordre envoyé au broker.

Cela permet de développer et valider tout le système sans risque d'exécution involontaire.

---

# 4. Real-Time Market Data

## 4.1 Fournisseur initial

Le fournisseur initial prévu est **Alpaca**.

Le flux WebSocket peut fournir notamment :

* trades ;
* quotes ;
* bars ;
* news.

Architecture :

```text
Alpaca WebSocket
       │
       ▼
 MarketDataFeed
       │
       ▼
    EventBus
```

L'utilisation d'un WebSocket est privilégiée à du polling périodique.

---

## 4.2 Types d'événements

Le système doit manipuler des événements génériques :

```python
MarketEvent
TradeEvent
QuoteEvent
BarEvent
NewsEvent
FundamentalEvent
PortfolioEvent
RiskEvent
DecisionEvent
```

Chaque événement doit posséder au minimum :

```python
timestamp
symbol
event_type
source
payload
```

---

# 5. Event Bus

Le `EventBus` est le cœur logiciel permettant de découpler les différents modules.

Exemple :

```text
MarketDataFeed
      │
      ▼
   EventBus
      │
      ├── Feature Engine
      ├── Portfolio Engine
      ├── Risk Engine
      ├── News Engine
      ├── HMM
      └── Storage
```

Les composants ne doivent pas directement dépendre les uns des autres.

Exemple :

```python
await event_bus.publish(event)
```

Puis :

```python
@event_bus.subscribe("trade")
async def on_trade(event):
    ...
```

Cela permet d'ajouter ultérieurement :

* nouvelles sources de données ;
* nouveaux modèles ;
* nouveaux brokers ;
* nouveaux dashboards ;

sans réécrire le moteur central.

---

# 6. Multi-Time-Horizon Architecture

Le système doit analyser plusieurs horizons simultanément.

## High Frequency / Fast

Exemples :

```text
tick
5 min
30 min
```

Features :

* returns ;
* spread ;
* volume ;
* short momentum ;
* short volatility ;
* mean reversion.

---

## Short Term

```text
1h
4h
1d
```

Features :

* momentum ;
* volatility ;
* correlation ;
* volume ;
* regime.

---

## Medium Term

```text
5d
20d
60d
```

Features :

* trend ;
* momentum ;
* volatility ;
* relative strength ;
* fundamental reaction.

---

## Long Term

```text
120d
252d+
```

Features :

* long-term momentum ;
* drawdown ;
* fundamentals ;
* valuation ;
* structural regime.

---

# 7. Important : les données n'ont pas toutes la même fréquence

Chaque feature doit avoir son propre rythme d'actualisation.

```text
Tick
 │
 ├── price
 ├── spread
 └── microstructure

5 min
 │
 ├── volatility
 ├── momentum
 └── HMM-HF

1 hour
 │
 ├── medium momentum
 ├── correlation
 └── HMM-MT

1 day
 │
 ├── long momentum
 ├── drawdown
 └── HMM-LT

Event
 │
 ├── earnings
 ├── guidance
 ├── news
 └── filings
```

Il ne faut donc **jamais recalculer un modèle lent inutilement à chaque tick**.

---

# 8. Online Learning

## 8.1 Principe

Le système est initialisé avec un petit historique de bootstrap puis fonctionne en mode online.

```text
Historical Seed
       │
       ▼
Initial Model
       │
       ▼
REAL-TIME
       │
       ├── inference
       │
       └── online update
```

Le but n'est pas :

```text
historique → entraînement → modèle figé
```

mais :

```text
historique minimal
      ↓
modèle initial
      ↓
observation
      ↓
prediction
      ↓
nouvelle observation
      ↓
model update
      ↓
prediction améliorée
      ↓
...
```

---

# 9. HMM Online

Le HMM sert principalement à détecter les régimes.

Exemples de régimes :

```text
TREND_LOW_VOL
TREND_HIGH_VOL
SIDEWAYS
HIGH_VOL
CRISIS
```

Le modèle ne doit pas simplement retourner :

```text
regime = CRISIS
```

Il doit conserver une distribution :

```text
TREND_LOW_VOL     0.72
SIDEWAYS          0.18
HIGH_VOL          0.08
CRISIS            0.02
```

Ainsi les décisions peuvent prendre en compte l'incertitude.

---

## 9.1 Observation vector

Exemple :

$$
x_t =
[
r_t,
\sigma_t,
volume_t,
momentum_t,
correlation_t,
drawdown_t
]
$$

Le modèle estime :

$$
P(S_t \mid x_{1:t})
$$

---

## 9.2 HMM par horizon

On prévoit plusieurs HMM :

```text
HMM-HF
    5m / 30m

HMM-MT
    1h / 4h

HMM-LT
    1d
```

Exemple :

```text
HF:
    trend       72%
    sideways    21%
    crisis       7%

MT:
    trend       81%
    sideways    14%
    crisis       5%

LT:
    trend       64%
    sideways    29%
    crisis       7%
```

Le système peut donc détecter un désaccord entre horizons.

---

# 10. Online Volatility

Une première version utilise EWMA.

$$
\sigma_t^2 =
\lambda\sigma_{t-1}^2
+
(1-\lambda)r_t^2
$$

Avantages :

* aucun retraining ;
* très peu de calcul ;
* mise à jour en temps réel ;
* état compact.

---

# 11. Momentum

Plusieurs horizons :

```text
momentum_5m
momentum_30m
momentum_1h
momentum_1d
momentum_20d
momentum_60d
momentum_252d
```

Les signaux doivent idéalement être normalisés par la volatilité.

Exemple :

$$
M_{i,h} =
\frac{R_{i,h}}{\sigma_{i,h}}
$$

---

# 12. Mean Reversion

Principalement utile sur les horizons courts.

Exemples de features :

* distance à moyenne mobile ;
* z-score ;
* short-term reversal ;
* distance à VWAP ;
* deviation from expected range.

---

# 13. Correlation Engine

Le système maintient une matrice :

$$
\Sigma_t
$$

ou une matrice de covariance/volatilité équivalente.

Elle sert à calculer :

* portfolio volatility ;
* diversification ;
* concentration ;
* risk contribution ;
* correlation spikes.

---

# 14. Portfolio State

Chaque position possède un état complet.

Exemple :

```python
PositionState(
    symbol="SPY",

    quantity=10,
    price=650.20,
    avg_price=620.00,

    market_value=6502,
    pnl=302,

    weight=0.21,
    target_weight=0.18,

    volatility=0.17,
    risk_contribution=0.23,

    signal_hf=0.42,
    signal_st=0.61,
    signal_mt=0.73,
    signal_lt=0.81,

    regime="TREND_LOW_VOL",
    regime_probability=0.82,
)
```

---

# 15. Risk Engine

Le Risk Engine doit être indépendant des modèles de signal.

Il surveille notamment :

* position limits ;
* weight limits ;
* target drift ;
* volatility ;
* portfolio volatility ;
* drawdown ;
* concentration ;
* correlation ;
* leverage ;
* risk contribution ;
* signal reversal ;
* regime changes ;
* signal disagreement.

---

# 16. Risk Contribution

Pour une covariance \(\Sigma\) et des poids \(w\) :

$$
\sigma_p =
\sqrt{w^T\Sigma w}
$$

Contribution marginale :

$$
MRC_i =
\frac{(\Sigma w)_i}{\sigma_p}
$$

Risk contribution :

$$
RC_i =
w_i MRC_i
$$

Contribution relative :

$$
RC_i^{relative}
=
\frac{w_i(\Sigma w)_i}
{w^T\Sigma w}
$$

Cela permet d'éviter de regarder uniquement le poids nominal.

Une position de 10 % peut représenter beaucoup plus que 10 % du risque du portefeuille.

---

# 17. Target Allocation

Le moteur doit distinguer :

```text
CURRENT WEIGHT
TARGET WEIGHT
DRIFT
```

Exemple :

```text
Current = 21%
Target  = 18%
Drift   = +3%
```

Le target weight peut dépendre de :

```text
signal
regime
volatility
correlation
risk budget
portfolio state
qualitative score
```

---

# 18. Risk Parity / HRP

Une méthode possible pour transformer les signaux en allocation est :

```text
Signal
   ↓
Risk Budget
   ↓
HRP / Risk Allocation
   ↓
Target Weights
```

Le système doit pouvoir utiliser :

* volatility targeting ;
* risk parity ;
* hierarchical risk parity ;
* concentration constraints.

---

# 19. Qualitative / Fundamental Engine

Le système ne doit pas être purement technique.

Il doit également intégrer les informations concernant les entreprises.

Exemples :

### Financial performance

```text
Revenue growth
EPS growth
Free cash flow growth
Margin change
Debt change
ROIC
```

### Earnings

```text
earnings surprise
EPS surprise
revenue surprise
guidance
guidance revision
```

### Valuation

```text
P/E
EV/EBITDA
P/S
FCF yield
```

### Analyst / market information

```text
estimate revisions
earnings revisions
consensus changes
```

---

# 20. Fundamental data est event-driven

Une donnée trimestrielle ne doit pas être considérée comme une feature realtime classique.

Exemple :

```text
Quarterly Earnings
       ↓
Fundamental Event
       ↓
Feature Update
       ↓
Online Model Update
```

Une donnée fondamentale doit conserver :

```text
period
publication_time
available_time
source
value
```

---

# 21. Information timing

C'est une contrainte critique.

Il faut distinguer :

```text
period
    ↓
quand la donnée concerne

publication_time
    ↓
quand elle est publiée

received_time
    ↓
quand notre système la reçoit

decision_time
    ↓
quand le modèle l'utilise
```

Le moteur ne doit jamais utiliser une information avant sa disponibilité réelle.

Cela évite le look-ahead bias.

---

# 22. News Engine

Les news sont traitées comme des événements.

Pipeline :

```text
News
 ↓
AI / NLP
 ↓
Structured Extraction
 ↓
Validation
 ↓
Quantitative Features
 ↓
Online Model
```

Exemples de catégories :

```text
earnings
guidance
product
management
M&A
regulation
litigation
macro
other
```

---

# 23. Structured AI Output

Le LLM ne doit pas produire une dissertation.

Il doit transformer :

```text
article / filing / news
```

en :

```text
structured financial event
```

Exemple conceptuel :

```json
{
  "symbol": "XXX",
  "event_type": "earnings",
  "sentiment": 0.82,
  "relevance": 0.97,
  "novelty": 0.91,
  "revenue_growth": 0.18,
  "earnings_surprise": 0.07,
  "guidance_change": 0.65,
  "risk_impact": -0.12,
  "horizon_days": 60,
  "confidence": 0.96
}
```

Les Structured Outputs / JSON Schema permettent de contraindre directement la sortie du modèle à une structure définie.

---

# 24. LLM ≠ Decision Maker

Le LLM ne doit pas faire :

```text
article
 ↓
BUY
```

Il doit faire :

```text
article
 ↓
structured information
 ↓
features
 ↓
online quantitative model
 ↓
signal
```

Le LLM sert principalement de :

> **text → structured financial information converter**

---

# 25. Novelty

Le système doit éviter de compter plusieurs fois la même information.

Exemple :

```text
Article 1:
"NVIDIA raises guidance"

Article 2:
"NVIDIA outlook improves"

Article 3:
"NVIDIA raises forecast"
```

Ces articles peuvent représenter le même événement.

Le moteur doit donc estimer :

```text
novelty
event_similarity
event_relevance
```

Un événement déjà connu ne doit pas générer artificiellement trois nouveaux signaux.

---

# 26. Confidence

Chaque extraction AI doit comporter une confiance :

```text
confidence = 0.94
```

Cette confiance peut ensuite pondérer l'information :

$$
S_\text{event}
=
confidence
\times
relevance
\times
novelty
\times
impact
$$

---

# 27. Qualitative Score

Une première représentation peut être :

$$
Q_i =
\sum_k w_k f_{i,k}
$$

avec :

```text
earnings
revenue growth
margin
FCF
guidance
news sentiment
analyst revisions
balance sheet
...
```

Mais les poids ne doivent pas nécessairement rester fixes.

Le système doit pouvoir apprendre progressivement leur pouvoir prédictif.

---

# 28. Online Predictive Learning

Pour chaque facteur, on peut mesurer sa relation avec les rendements futurs :

$$
E[r_{t+h}|X_t]
$$

pour plusieurs horizons \(h\).

Exemple :

```text
                    Earnings Surprise
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             +1d          +5d          +20d
              │            │            │
            return       return       return
```

Le système peut ainsi apprendre qu'une feature est :

```text
short-term
medium-term
long-term
```

---

# 29. Signal Fusion

Le signal final ne doit pas être uniquement basé sur le prix.

On a :

```text
Market Signal
Regime Signal
Fundamental Signal
Event Signal
```

Pour chaque actif \(i\) et horizon \(h\) :

$$
S_{i,h}
=
F(
S^{market}_{i,h},
S^{regime}_{i,h},
S^{fundamental}_{i,h},
S^{event}_{i,h}
)
$$

La fonction \(F\) peut progressivement devenir un modèle appris.

---

# 30. Signal Disagreement

Un indicateur important est le désaccord entre horizons.

Exemple :

```text
HF      +0.72
ST      +0.41
MT      -0.18
LT      -0.43
```

Le système doit identifier :

```text
SHORT TERM BULLISH
LONG TERM BEARISH
```

Cela peut être une information de risque à part entière.

---

# 31. Decision Engine : UREBALANCE / UDONOTHING

Le Decision Engine combine :

```text
Signals
+
Regime
+
Portfolio
+
Risk
+
Target weights
```

Il n'y a pas :

```text
BUY
SELL
HOLD
```

Il n'y a que :

```text
UREBALANCE
UDONOTHING
```

Le moteur compare deux utilités : $U_{rebalance}$ et $U_{donothing}$.

### `UDONOTHING`

On accepte la situation actuelle malgré le drift.

Coûts potentiels :

```text
risk drift
concentration
higher portfolio volatility
loss of diversification
```

### `UREBALANCE`

On revient vers la cible, mais on prend en compte :

```text
transaction costs
spread
slippage
market impact
tax
execution risk
```

Conceptuellement :

$$
U_{rebalance} = Benefit_{risk} - Cost_{execution}
$$

et :

$$
Decision =
\begin{cases}
UREBALANCE & \text{si bénéfice > coût} \\
UDONOTHING & \text{sinon}
\end{cases}
$$

---

## 31.1 Rebalancement partiel

`UREBALANCE` ne signifie pas forcément `10% → 15%` immédiatement.

Le moteur peut décider une trajectoire :

```text
10%
 ↓
11.5%
 ↓
13%
 ↓
15%
```

Donc :

```python
Decision(
    symbol="AAPL",
    action="UREBALANCE",
    current_weight=0.10,
    target_weight=0.15,
    execution_weight=0.115,
    urgency=0.72,

    reason=[
        "target drift",
        "risk contribution elevated",
        "medium-term signal strengthened"
    ],
)
```

La décision ne devient pas automatiquement un ordre.

---

# 32. Execution Engine

Appelé **seulement après `UREBALANCE`**.

Il reçoit :

```text
current weight
target weight
execution weight
```

et cherche **comment** exécuter.

### Fill probability

$$
P_{fill}(p, q, \Delta t)
$$

### Execution cost

$$
C(q) = C_{spread} + C_{slippage} + C_{impact} + C_{fees}
$$

### Participation

$$
participation = \frac{Q_{order}}{V_{market}}
$$

### Signal half-life

Un signal HF qui disparaît dans 20 minutes n'est pas exécuté comme un signal
LT valable plusieurs mois : l'urgence dépend de la demi-vie du signal.

---

# 33. Order Optimizer

Il peut optimiser :

```text
quantity
limit price
aggressiveness
timing
execution duration
```

Conceptuellement :

$$
U(o) = P_{fill}(o)\,E[\alpha(o)] - C_{execution}(o) - C_{risk}(o)
$$

puis :

$$
o^* = \arg\max_o U(o)
$$

Le système peut donc produire :

```text
UREBALANCE

AAPL
10% → 15%

execute:
+1.5% now
limit = 249.99
urgency = 0.72
expected fill = 72%
```

---

# 34. Deux boucles d'apprentissage

### Alpha loop

```text
Market
 ↓
Signal
 ↓
Prediction
 ↓
Realized return
 ↓
Model update
```

### Execution loop

```text
Order proposal
 ↓
Execution
 ↓
Fill / partial fill / no fill
 ↓
Realized slippage
 ↓
Market impact
 ↓
Model update
```

Le système apprend donc deux choses différentes :

$$
\boxed{\text{Quel target weight ?}}
\qquad
\boxed{\text{Comment atteindre ce target ?}}
$$

---

# 35. Alerts

Exemples :

```text
POSITION_DRIFT
VOLATILITY_SPIKE
CORRELATION_SPIKE
DRAWDOWN
RISK_LIMIT
REGIME_CHANGE
SIGNAL_REVERSAL
SIGNAL_DISAGREEMENT
NEW_EARNINGS
GUIDANCE_CHANGE
IMPORTANT_NEWS
```

---

# 36. Dashboard

Le dashboard doit permettre de voir en temps réel :

```text
PORTFOLIO
────────────────────────────

Equity
Cash
Portfolio volatility
Drawdown
Leverage

RISK
────────────────────────────

AAPL       RC 23%
MSFT       RC 18%
SPY        RC 31%

REGIME
────────────────────────────

HF   TREND_LOW_VOL  72%
MT   SIDEWAYS       54%
LT   HIGH_VOL       41%

SIGNALS
────────────────────────────

AAPL
HF   +0.72
ST   +0.41
MT   -0.18
LT   -0.43

ALLOCATION
────────────────────────────

AAPL
Current    10%
Target     15%
Drift       5%

DECISION
────────────────────────────

UREBALANCE

EXECUTION
────────────────────────────

Expected fill    72%
Expected impact   0.04%
Urgency           0.71

QUALITATIVE
────────────────────────────

Fundamental score
News score
Latest events
Confidence
```

---

# 37. Architecture logicielle

```text
trading_engine/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── market_feed.py
│   ├── news_feed.py
│   ├── fundamental_feed.py
│   ├── event_bus.py
│   ├── market_state.py
│   └── bar_builder.py
│
├── portfolio/
│   ├── positions.py
│   └── portfolio.py
│
├── features/
│   ├── volatility.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── correlations.py
│   └── fundamentals.py
│
├── models/
│   ├── hmm.py
│   ├── online_regression.py
│   ├── online_factors.py
│   └── regime.py
│
├── ai/
│   ├── structured_extraction.py
│   ├── news_classifier.py
│   └── schemas.py
│
├── signals/
│   ├── market.py
│   ├── fundamental.py
│   ├── event.py
│   └── fusion.py
│
├── risk/
│   ├── portfolio_risk.py
│   ├── position_risk.py
│   ├── covariance.py
│   └── limits.py
│
├── allocation/
│   ├── risk_parity.py
│   ├── hrp.py
│   └── targeting.py
│
├── decision/
│   └── rebalance.py
│
├── execution/
│   ├── fill_model.py
│   ├── cost_model.py
│   ├── impact_model.py
│   ├── order_pricer.py
│   └── optimizer.py
│
├── storage/
│   ├── database.py
│   └── event_log.py
│
├── api/
│   └── server.py
│
├── dashboard/
│
├── tests/
│
└── main.py
```

---

# 38. Core data model

Chaque objet important doit être immutable ou versionné autant que possible.

### Position

```python
Position(
    symbol,
    quantity,
    avg_price
)
```

### Market State

```python
MarketState(
    symbol,
    price,
    timestamp
)
```

### Signal

```python
Signal(
    symbol,
    horizon,
    value,
    confidence,
    timestamp,
    source
)
```

### Fundamental Event

```python
FundamentalEvent(
    symbol,
    event_type,
    value,
    period,
    available_at,
    source
)
```

### News Event

```python
NewsEvent(
    symbol,
    event_type,
    sentiment,
    relevance,
    novelty,
    confidence,
    timestamp,
    source
)
```

### Risk State

```python
RiskState(
    weight,
    target_weight,
    volatility,
    risk_contribution,
    drawdown
)
```

### Decision

```python
Decision(
    decision_id,
    symbol,
    action,            # UREBALANCE | UDONOTHING
    current_weight,
    target_weight,
    execution_weight,
    urgency,
    reason,
    timestamp
)
```

### Order Proposal

```python
OrderProposal(
    decision_id,
    symbol,
    quantity,
    limit_price,
    urgency,
    expected_fill,
    expected_cost,
    timestamp
)
```

---

# 39. Asynchronous architecture

Le système doit être conçu autour de `asyncio`.

Exemple :

```python
async def main():

    await asyncio.gather(
        market_feed.run(),
        news_feed.run(),
        portfolio_monitor.run(),
        risk_engine.run(),
        dashboard.run(),
    )
```

Chaque composant communique via l'EventBus.

---

# 40. Exemple de boucle complète

```python
async for event in feed:

    await event_bus.publish(event)

    if event.type == "TRADE":
        market_state.update(event)

        features.update(event)

        hmm_hf.update(features.fast)

        portfolio.update_market(event)

        risk.update(portfolio)

    if event.type == "BAR_1H":
        hmm_mt.update(features.medium)

    if event.type == "BAR_1D":
        hmm_lt.update(features.long)

    if event.type == "NEWS":
        structured_event = await ai.extract(event)

        qualitative.update(structured_event)

        online_model.update(structured_event)

    decision = decision_engine.evaluate()

    if decision.action == "UREBALANCE":
        proposal = execution_engine.optimize(decision)
        await event_bus.publish(proposal)
```

---

# 41. Storage

Même si le moteur est realtime, **tout doit être enregistré**.

À conserver :

```text
raw market events
features
model states
HMM probabilities
fundamental events
news
AI outputs
signals
risk states
target weights
decisions (UREBALANCE / UDONOTHING)
order proposals
fills / slippage
```

Cela permet ensuite de répondre à :

> "Pourquoi le modèle voulait réduire cette position à 14:32 ?"

---

# 42. Reproductibilité

Chaque décision doit pouvoir être reconstruite à partir de :

```text
timestamp
market state
features
model version
model parameters
signals
risk state
fundamental information available at that time
news available at that time
```

Un `decision_id` doit idéalement relier tout cela.

---

# 43. Backtesting

Le backtesting n'est pas le moteur principal.

Il sert à :

* validation ;
* analyse historique ;
* comparaison de modèles ;
* détection de bugs ;
* mesure de performance ;
* ablation studies ;
* analyse des facteurs.

Le moteur live reste :

```text
online inference
+
online learning
```

Le log historique permet ensuite de reconstruire le comportement passé.

## Event replay : un seul moteur

Un backtest classique (`for day in data: strategy(day)`) ne reproduit ni
l'ordre des événements, ni l'arrivée des news, ni l'état des modèles online.

Le moteur live et le moteur de validation sont donc **le même code** ; seule
la source d'événements change :

```text
                    EVENT SOURCE
                         │
              ┌──────────┴──────────┐
              │                     │
            LIVE                 REPLAY
      (Alpaca, simulé)     (journal enregistré)
              │                     │
              └──────────┬──────────┘
                         ▼
                    SAME ENGINE
                         │
                         ▼
                   SAME DECISIONS
```

Chaque événement brut reçu en live est écrit dans un journal (JSONL) dans
l'ordre de réception ; le `ReplayFeed` le relit dans le même ordre. Le moteur
n'utilise jamais l'heure système dans sa logique : l'horloge est celle des
événements. Rejouer un journal redonne donc exactement le même état.

---

# 44. Anti-look-ahead

Le système doit respecter strictement :

$$
information\_available(t)
$$

Une information avec :

```text
period = Q2
publication = 2026-08-10
```

ne peut pas être utilisée pour une décision du :

```text
2026-07-20
```

Même si la base de données historique contient déjà la valeur finale.

---

# 45. Initialisation

Le système ne démarre pas totalement "from zero".

Pour les modèles nécessitant une distribution initiale :

```text
historical seed
        ↓
initial parameters
        ↓
online adaptation
```

Cela concerne notamment :

* HMM ;
* covariance ;
* volatility ;
* normalisation des features.

---

# 46. Online vs retraining

Tous les modèles ne doivent pas être entraînés à la même fréquence.

| Modèle                 | Update            |
| ---------------------- | ----------------- |
| Price state            | tick              |
| EWMA volatility        | tick/bar          |
| Momentum               | bar               |
| Correlation            | 5m/1h             |
| HMM-HF                 | 5m                |
| HMM-MT                 | 1h                |
| HMM-LT                 | 1d                |
| Fundamentals           | event             |
| News NLP               | event             |
| Long-term factor model | daily             |
| Allocation             | event / scheduled |

---

# 47. AI Architecture

Le LLM doit être utilisé principalement pour les données non structurées :

```text
News
Filings
Earnings calls
Management statements
```

Pipeline :

```text
RAW TEXT
   ↓
LLM
   ↓
STRUCTURED OUTPUT
   ↓
VALIDATION
   ↓
FEATURE STORE
   ↓
ONLINE MODEL
```

Le modèle quantitatif reste responsable de l'interprétation statistique.

---

# 48. Modèle AI économique

Il n'est pas nécessaire d'utiliser un gros modèle pour chaque événement.

Architecture :

```text
                New event
                    │
                    ▼
             Cheap / fast model
                    │
             ┌──────┴──────┐
             │             │
         confident       uncertain
             │             │
             ▼             ▼
          accept       stronger model
```

Cela permet de réduire :

* latence ;
* coût ;
* consommation de tokens.

---

# 49. Limites et garde-fous

L'architecture empile beaucoup de modèles
(`data → features → HMM → signals → fusion → risk → allocation → decision → execution`).
Chaque étage ajoute de l'incertitude : un petit avantage statistique initial
peut disparaître en bout de chaîne. Les garde-fous suivants font partie du
design, pas d'une optimisation ultérieure.

## 49.1 Non-stationnarité

Une relation $X_t \rightarrow r_{t+1}$ apprise aujourd'hui peut disparaître
demain (concept drift, régimes, volatilité, corrélations, microstructure).
L'apprentissage online peut aussi **apprendre du bruit** et rendre le modèle
instable. Chaque modèle online doit avoir :

```text
forgetting factor
regularization
minimum sample size
update frequency
stability / drift monitoring
```

## 49.2 Complexité et valeur incrémentale

`complexity ⇏ performance`. Une stratégie de référence
**momentum simple + volatility targeting** sert de baseline. Chaque module
peut être désactivé dans la configuration et sa **valeur incrémentale** est
mesurée en replay contre cette baseline ; un module qui n'apporte rien est
retiré.

## 49.3 Le HMM est une estimation, pas une vérité

Le HMM fournit une estimation probabiliste d'un état latent, utilisée comme
feature de régime. Il ne « sait » pas dans quel marché nous sommes.

## 49.4 Double comptage

Earnings surprise, sentiment LLM et momentum peuvent venir **du même
événement**. Additionner leurs signaux compte trois fois une seule
information. À gérer :

```text
event identity
event clustering
correlation between signals
novelty
information overlap
```

## 49.5 Le LLM est une source d'incertitude

`JSON valide ≠ information correcte ≠ information utile`. Pipeline de
validation :

```text
LLM
 ↓
schema validation
 ↓
range validation
 ↓
source validation
 ↓
event deduplication
 ↓
confidence
 ↓
impact threshold
```

## 49.6 Stabilité de la cible : hystérésis

Un signal instable fait osciller la cible (15 % → 14 % → 16 % → 13 %).
`UREBALANCE` n'est envisagé que si

$$
|w_{target} - w_{current}| > \epsilon
$$

(`10 % → 10.8 %` : `UDONOTHING` ; `10 % → 15 %` : candidat `UREBALANCE`).

## 49.7 `UREBALANCE` n'est pas une boîte noire

Les décisions restent strictement séparées :

```text
Decision:    UREBALANCE
Allocation:  target_weight = 15%
Execution:   current = 10%, execution_target = 12%
Order:       quantity, limit price, timing
```

## 49.8 Confiance du signal ≠ confiance d'exécution

Au départ il n'y a presque aucune donnée d'exécution propriétaire : les
modèles de fill et de slippage reposent sur des hypothèses. On distingue
`MODEL CONFIDENCE` et `EXECUTION CONFIDENCE`, pour éviter qu'un signal très
confiant masque un modèle d'exécution très incertain.

## 49.9 Alpha net et significativité économique

$$
\alpha_{net} = \alpha_{gross} - C_{spread} - C_{slippage} - C_{impact} - C_{fees} - C_{tax}
$$

Un alpha de 0.15 % est inutile si l'exécution coûte 0.20 %. Les signaux sont
donc exprimés **en rendement attendu**, directement comparable aux coûts,
et pas en score sans unité.

## 49.10 Incertitude explicite

`+2.4 % ± 0.3 %` et `+2.4 % ± 5 %` ne sont pas la même information. Chaque
prédiction porte une distribution :

$$
r_{future} \sim \mathcal{D}(\mu, \sigma)
$$

```python
Signal(symbol, horizon, mean, std, n_obs, timestamp, source)
```

## 49.11 Timing

Les données ont des fréquences très différentes (ms pour les ticks, trimestre
pour les fondamentaux) et arrivent avec des délais. Chaque étape est horodatée :

```text
event_time
received_time
processing_time
decision_time
execution_time
```

pour ne jamais utiliser une information qu'on n'aurait pas eue au moment de
la décision.

## 49.12 Le portefeuille est un système couplé

Modifier une position change le risque de tout le portefeuille
($\sigma_p=\sqrt{w^T\Sigma w}$). Quatre signaux excellents sur des actifs
exposés au même facteur ne font pas quatre paris. L'allocation reste
**portfolio-level**.

## 49.13 Attribution

Le système doit pouvoir expliquer chaque changement de cible :

```text
MSFT target 8% → 11%

+2.1%  medium-term momentum
+1.2%  earnings revision
+0.8%  regime
-0.6%  portfolio concentration
-0.4%  volatility
-0.2%  correlation
--------------------------------
+2.9%
```

La cible est donc construite comme une somme de contributions traçables.

## 49.14 Feedback loops

`signal → trade → mouvement de prix → nouveau signal` : le système peut
réagir à ses propres actions. Les données d'apprentissage distinguent le
mouvement dû au marché du mouvement induit par l'exécution.

## 49.15 Constraint Engine

Entre allocation et décision, un module explicite rend la cible réalisable :

```text
max position
max sector exposure
max turnover
max leverage
min cash
max tracking error
max portfolio volatility
max execution participation
tax constraints
```

## 49.16 Priorités

| Priorité | Problème                         | Solution                                        |
| -------- | -------------------------------- | ----------------------------------------------- |
| 1        | Non-stationnarité                | drift detection + forgetting + model monitoring |
| 2        | Overfitting / complexité         | ablation + valeur incrémentale                  |
| 3        | Double comptage                  | event clustering + corrélation des signaux      |
| 4        | Incertitude d'exécution          | modèles fill / slippage + confidence            |
| 5        | Validation realtime              | event replay engine                             |

---

# 50. Robustness / Adversarial Defense

Objectif : **résister à la manipulation, aux données trompeuses et à
l'exploitation du comportement du système**. Il ne s'agit pas de dissimuler
une stratégie ni de contourner la surveillance du marché.

Principe clé : **aucune source et aucun modèle ne peut, seul, provoquer un
gros `UREBALANCE`**. La question n'est pas « la prédiction est-elle bonne ? »
mais :

> **La prédiction est-elle stable, indépendante, calibrée et économiquement utile ?**

`UDONOTHING` devient alors aussi la réponse normale quand l'information est
trop incertaine ou trop contradictoire, pas seulement quand la cible n'a pas
bougé.

## 50.1 Data Integrity

Avant tout modèle. Un saut $|r_t| > k\sigma$ n'est pas automatiquement un
crash : mauvaise donnée, split, corporate action, glitch de flux, timestamp
incorrect.

```text
timestamp dans le futur / hors ordre
prix ou taille invalides
saut de prix non confirmé      → quarantaine jusqu'au trade suivant
saut confirmé à un ratio de split (2:1, 3:1, 1:2…) → corporate action probable
quote croisée (bid > ask), spread anormal
barre incohérente (high < low, close hors range)
flux figé (plus de données alors que le marché vit)
```

Un saut isolé est mis en **quarantaine** : confirmé par le trade suivant, il
est accepté ; démenti (retour à l'ancien niveau), il est rejeté comme glitch.
Chaque symbole et chaque source a un `DataIntegrityScore`.

## 50.2 Safety Engine et hard controls

```text
NORMAL     tout est cohérent
DEGRADED   désaccord des modèles, qualité des données ↓, modèle dégradé,
           corporate action → rebalancements réduits, symboles concernés gelés
HALTED     flux corrompu, explosion numérique, perte journalière max,
           rejets répétés des hard controls → uniquement UDONOTHING
```

- l'escalade est immédiate, le retour de DEGRADED à NORMAL demande plusieurs
  évaluations saines consécutives ;
- **HALTED exige une remise en route manuelle** ;
- **HALTED ne signifie pas liquider** : on arrête de décider, on ne vend pas
  dans la panique.

Les **hard controls** sont distincts du Constraint Engine : le Constraint
Engine façonne la cible ; les hard controls sont un **veto final** sur les
cibles et les ordres, avec leurs propres limites que les modèles ne peuvent
pas modifier (poids maximal, exposition, taille d'ordre, participation, collar
de prix, nombre d'opérations par jour, turnover journalier). C'est la même
logique que les contrôles pré-trade d'accès au marché : bloquer les ordres
erronés ou hors limites, même si un modèle produit `target = 0.99`.

## 50.3 Model ensemble : accord ≠ indépendance

L'accord entre modèles n'a de valeur que si leurs **erreurs** sont
indépendantes. HMM, TSFM et momentum lisent la même série de prix : leur
accord est gonflé par construction. La fusion utilise donc la **corrélation
des erreurs** mesurée hors échantillon (deux modèles corrélés à 0.9 valent
à peu près un seul modèle).

Le désaccord entre **horizons** reste une information, pas forcément un
défaut. Le HMM ne vote pas : il estime des régimes de volatilité et sert à
**conditionner** les autres modèles. Un TSFM n'est ajouté que s'il bat la
baseline en replay (§49.2).

## 50.4 Modèle dégradé

Quand les données deviennent incompatibles avec le modèle
($P(data \mid model) \ll$ niveau habituel), le système déclenche
`MODEL_DEGRADED` au lieu d'augmenter sa confiance : dérive de l'erreur
(Page-Hinkley) pour les modèles prédictifs, chute durable de la
log-vraisemblance pour les HMM.

## 50.5 Prédictibilité ≠ fiabilité

- **prédiction** : $\mu \pm \sigma$ ;
- **fiabilité** : le modèle a-t-il été fiable récemment, **dans ce contexte** ?

La fiabilité est d'abord le skill hors échantillon par régime, niveau de
volatilité et horizon, ramené vers le skill global tant que les données sont
rares (un méta-modèle complet demande beaucoup de résultats observés).

## 50.6 Stress tests et perturbations

- **online**, au moment d'un rebalancement : recalculer la cible sous
  plusieurs scénarios (vol ×2, corrélations ↑, rendement −2σ, spread ×3) ; une
  cible qui s'effondre sous des hypothèses proches n'est pas robuste ;
- **offline**, dans les tests et le replay : de petites perturbations
  $\delta$ des données ne doivent pas changer la décision,
  $\|\Delta S\| \ll \|\delta\|$.

## 50.7 News

Articles quasi identiques → **un seul événement** (clustering), et

$$
Signal_{news} = Impact \times Confidence \times Novelty \times SourceQuality
$$

(source primaire ≠ quinze reprises d'une même rumeur).

## 50.8 Décision : unités économiques, pas soupe de scores

La décision garde tous les diagnostics pour l'explication :

```python
Decision(
    action="UREBALANCE",
    target_weight=0.15,
    expected_return=0.024, uncertainty=0.011,
    model_agreement=0.82, model_reliability=0.76,
    data_quality=0.97, source_quality=0.91,
    robustness_score=0.88, portfolio_risk=0.12,
    execution_confidence=0.74,
)
```

mais **ne multiplie pas** ces scores entre eux (ils ne sont pas calibrés sur
la même échelle et leur produit tend vers zéro arbitrairement). Chacun agit
en unités économiques :

- **filtres** : qualité des données, état de sécurité, instabilité aux stress
  tests → `UDONOTHING` ;
- **incertitude** : désaccord et faible fiabilité **élargissent** $\sigma$ ;
- **règle** :

$$
UREBALANCE \iff Benefit - Cost_{execution} > k \cdot \sigma_{effective}
$$

---

# 51. Fiscalité : profils par pays

Les taxes font partie des coûts d'un rebalancement
($\alpha_{net} = \alpha_{gross} - \dots - C_{tax}$) : un signal faible peut
être détruit par la taxe sur les transactions ou par l'impôt sur une
plus-value réalisée.

Chaque pays est décrit par un fichier TOML (`config/taxes/<PAYS>.toml`) ; le
code est générique et ne contient aucun taux :

```text
[meta]              pays, devise, sources, date de vérification
[regions]           groupes de pays (ex. EEA) utilisables dans les règles
[transaction_tax]   règles ordonnées (première correspondance) avec taux et plafond
[income_tax]        dividendes, intérêts, retenues étrangères
[capital_gains]     taux, date d'entrée en vigueur, exonération, report, step-up
[account_tax]       taxe annuelle sur la valeur d'un compte
```

Les instruments sont classés dans `config.yaml` (`asset_class`, `domicile`,
`distribution`, `registered_locally`) ; un instrument non classé reçoit la
règle par défaut (prudente) et déclenche un avertissement.

### Belgique (`config/taxes/BE.toml`)

| Taxe | Règle modélisée |
| ---- | --------------- |
| TOB  | 0,12 % ETF domiciliés EEE et obligations (plafond 1 300 €), 0,35 % actions et ETF hors EEE (plafond 1 600 €), 1,32 % fonds de capitalisation enregistrés en Belgique (plafond 4 000 €), à l'achat et à la vente ; auto-déclarée avec un broker étranger |
| Précompte mobilier | 30 % sur dividendes et intérêts, après retenue étrangère (US 15 %) ; exonération des premiers dividendes via la déclaration |
| Plus-values (2026) | 10 % sur les plus-values réalisées nettes de l'année, exonération annuelle de 10 000 € avec report de 1 000 €/an (5 ans max), plus-values historiques gelées au 31/12/2025, lots FIFO |
| Comptes-titres | 0,15 % au-delà d'une valeur moyenne de 1 M€ |

⚠ Le profil n'est **pas un conseil fiscal** : montants indexés et taxe sur
les plus-values récente, à vérifier auprès du SPF Finances puis à marquer
`verified_on`. Un trading très fréquent peut aussi être requalifié en revenus
divers (33 %).

Le moteur :

- comptabilise la TOB et les lots fiscaux à chaque fill ;
- estime **avant** de décider le coût fiscal de rejoindre la cible
  (TOB + impôt marginal sur les plus-values), qui entrera dans
  $U_{rebalance}$ (Decision Engine).

---

# 52. V1 Development Roadmap

## Phase 1 — Core

```text
[x] Project structure
[x] Configuration
[x] EventBus
[x] MarketState
[x] PositionState
[x] PortfolioState
```

## Phase 2 — Realtime

```text
[x] Alpaca WebSocket
[x] Trade events
[x] Quote events
[x] Bar events
[x] Reconnection
[x] Heartbeat
[x] Event timestamps
```

## Phase 3 — Features

```text
[x] Returns
[x] Momentum
[x] EWMA volatility
[x] Mean reversion
[x] Correlation
```

## Phase 4 — Replay & Storage

```text
[x] Event log (journal JSONL des événements bruts)
[x] ReplayFeed (même moteur, source rejouée)
[x] Replay déterministe (live == replay)
[x] Baseline : momentum + volatility targeting
[x] Interrupteurs de modules (ablation)
```

## Phase 5 — Online Models

```text
[x] Signal = rendement attendu ± incertitude
[x] HMM-HF
[x] HMM-MT
[x] HMM-LT
[x] Online factor model
[x] Forgetting / regularization / minimum sample size
[x] Stability & drift monitoring
```

## Phase 6 — Risk

```text
[x] Portfolio volatility
[x] Covariance
[x] Risk contribution
[x] Drawdown
[x] Concentration
[x] Limits
```

## Phase 7 — Allocation & Constraints

```text
[x] Signal → target weight
[x] Volatility targeting
[x] Risk parity
[x] HRP
[x] Constraint engine
[x] Target attribution
[x] Drift monitoring
```

## Phase 8 — Data Integrity & Safety

```text
[x] Sanity checks (timestamps, prix, quotes, barres)
[x] Quarantaine des sauts non confirmés
[x] Détection de corporate actions probables
[x] Flux figé
[x] DataIntegrityScore par symbole
[x] Safety Engine (NORMAL / DEGRADED / HALTED)
[x] Perte journalière maximale
[x] Hard controls (cibles et ordres)
```

## Phase 9 — Robustness

```text
[ ] MODEL_DEGRADED (vraisemblance HMM)
[ ] Fiabilité par contexte (régime, volatilité, horizon)
[ ] Corrélation des erreurs entre modèles
[ ] Stress tests de la cible
[ ] Tests de perturbation (suite de tests / replay)
```

## Phase 10 — Qualitative

```text
[ ] Fundamental data
[ ] Earnings events
[ ] Guidance
[ ] SEC/filings
[ ] News feed
```

## Phase 11 — AI

```text
[ ] Structured output schema
[ ] News extraction
[ ] Event classification
[ ] Sentiment
[ ] Novelty
[ ] Confidence
[ ] Fundamental extraction
[ ] Validation pipeline (schema, range, source)
[ ] Event deduplication / clustering
[ ] Source quality / diversity
```

## Phase 12 — Decision Engine

```text
[ ] Signal fusion (sans double comptage)
[ ] Risk checks
[ ] Hystérésis
[x] Profils fiscaux par pays (TOML) — Belgique
[ ] Coût fiscal dans U_rebalance
[ ] U_rebalance vs U_donothing (alpha net, k · σ effectif)
[ ] Filtres : data quality, safety state, stress tests
[ ] Partial rebalance (execution weight, urgency)
[ ] Reason generation
[ ] Alerts
```

## Phase 13 — Execution

```text
[ ] Cost model (spread, slippage, fees)
[ ] Impact model / participation
[ ] Fill model
[ ] Execution confidence
[ ] Order pricer
[ ] Order optimizer
[ ] Order proposals
[ ] Execution feedback loop
```

## Phase 14 — Dashboard

```text
[ ] Portfolio overview
[ ] Position monitor
[ ] Signal monitor
[ ] Regime monitor
[ ] Risk monitor
[ ] News/events
[ ] Decision history
[ ] Execution monitor
```

---

# 53. Final target architecture

```text
                          REAL-TIME DATA
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    Market Data            Fundamentals               News
         └──────────────────────┼──────────────────────┘
                                ▼
                         DATA INTEGRITY
                  (sanity checks, quarantaine,
                   score de qualité par source)
                                │
                                ▼
                         FEATURE ENGINE
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    HMM REGIMES          PREDICTIVE MODELS         NLP / LLM
  (conditionnement)    (factor, TSFM optionnel)   (événements)
         └──────────────────────┼──────────────────────┘
                                ▼
                         MODEL ENSEMBLE
                     ┌──────────┴──────────┐
                     ▼                     ▼
                PREDICTION            RELIABILITY
                  μ ± σ           (skill par contexte)
                     └──────────┬──────────┘
                                ▼
                          SIGNAL FUSION
                  (corrélation des erreurs entre
                   modèles, pas de double comptage)
                                │
                                ▼
                       ROBUSTNESS ENGINE
              (stress tests, désaccord, dégradation)
                                │
                                ▼
                          RISK ENGINE
                                │
                                ▼
                       ALLOCATION ENGINE
                                │
                                ▼
                       CONSTRAINT ENGINE
                                │
                                ▼
                    UREBALANCE / UDONOTHING
                          │           │
                          ▼           └──► aucune action
                    EXECUTION ENGINE
                          │
                          ▼
                     HARD CONTROLS
            (limites indépendantes des modèles)
                          │
                          ▼
                    ORDER PROPOSAL
                          │
                          ▼
                Human / Broker Execution
                          │
                          ▼
                  EXECUTION OBSERVED
                ┌─────────┴─────────┐
                ▼                   ▼
          ALPHA LEARNING     EXECUTION LEARNING


  SAFETY ENGINE : NORMAL / DEGRADED / HALTED
  supervise toute la chaîne (données, modèles, risque, ordres) ;
  en HALTED, seul UDONOTHING est possible.
```

---

# 54. Design principles

Le projet doit respecter les principes suivants :

1. **Realtime first**
2. **Online learning first**
3. **Multi-horizon by design**
4. **Risk before execution**
5. **Signal ≠ order**
6. **LLM ≠ trading decision**
7. **Every event is timestamped**
8. **No look-ahead**
9. **All model states are observable**
10. **All decisions are reproducible**
11. **Qualitative information becomes structured data**
12. **Historical data is primarily used for initialization and validation**
13. **Models update at frequencies appropriate to their timescale**
14. **Portfolio risk is evaluated globally, not position-by-position only**
15. **Every target allocation must have an explainable reason**
16. **No BUY / SELL: only UREBALANCE or UDONOTHING**
17. **Deciding to rebalance and deciding how to execute are separate problems**
18. **Alpha and execution are learned by two separate loops**
19. **Live and replay run the exact same engine**
20. **Signals are expected returns with uncertainty, compared to costs**
21. **Every module must prove its incremental value against a simple baseline**
22. **No single source or model can trigger a large rebalance on its own**
23. **Model agreement only counts if model errors are independent**
24. **Hard controls are independent of models and cannot be changed by them**
25. **When in doubt, UDONOTHING: uncertainty is a reason not to act**

---

# 55. First implementation milestone

La première milestone concrète est volontairement petite :

```text
Alpaca WebSocket
       ↓
EventBus
       ↓
MarketState
       ↓
PositionState
       ↓
PortfolioState
       ↓
EWMA Volatility
       ↓
5m / 1h / 1d bars
       ↓
Basic HMM
       ↓
Risk Engine
       ↓
Console output
```

Une fois cette boucle stable, on ajoute :

```text
Fundamentals
      +
News
      ↓
Structured AI
      ↓
Qualitative Features
      ↓
Online Predictive Model
```

Puis seulement :

```text
Target Allocation
      ↓
UREBALANCE / UDONOTHING
      ↓
Execution Engine
      ↓
Order Proposal
      ↓
Dashboard
      ↓
Optional Broker Execution
```

La philosophie en une ligne :

```text
REAL-TIME DATA
      ↓
FEATURES
      ↓
ONLINE MODELS
      ↓
SIGNALS
      ↓
RISK
      ↓
TARGET ALLOCATION
      ↓
UREBALANCE / UDONOTHING
      ↓
EXECUTION OPTIMIZATION
      ↓
ORDER PROPOSAL
      ↓
OBSERVE RESULT
      ↓
ONLINE LEARNING
      └──────────────→
```

**Objectif final : construire une sorte de "control tower" quantitative du portefeuille : le système observe continuellement le marché, apprend progressivement, maintient une représentation probabiliste de l'état de chaque actif et du portefeuille, intègre les informations fondamentales et qualitatives, mesure le risque global et explique en temps réel pourquoi l'allocation cible évolue.**
