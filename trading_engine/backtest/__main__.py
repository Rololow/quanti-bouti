"""Backtest en ligne de commande.

    # 1. données (clés Alpaca dans l'environnement ; une seule fois)
    python -m trading_engine.backtest download --start 2018-01-01 --end 2026-09-30

    # 2. backtest : moteur (2 variantes x grille) + références + walk-forward
    python -m trading_engine.backtest run

    # sans clé : dataset synthétique pour vérifier la chaîne (aucun edge)
    python -m trading_engine.backtest synthetic --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from trading_engine.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, resolve_path

DEFAULT_DATASET = "data/backtest/bars_30m.jsonl"
DEFAULT_OVERLAY = PROJECT_ROOT / "config" / "backtest.yaml"
DEFAULT_GRID = ["decision.holding_period=20d,60d,120d"]


def _date(text: str) -> date:
    return date.fromisoformat(text)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="python -m trading_engine.backtest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--overlay", action="append", default=None,
                        help=f"surcouche de config (défaut {DEFAULT_OVERLAY.relative_to(PROJECT_ROOT)})")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="télécharge les barres Alpaca et les taux BCE")
    dl.add_argument("--start", type=_date, required=True)
    dl.add_argument("--end", type=_date, required=True)
    dl.add_argument("--symbols", help="liste séparée par des virgules (défaut : feed.symbols)")
    dl.add_argument("--timeframe", default="30m")
    dl.add_argument("--feed", default="sip", help="sip (historique complet) ou iex")
    dl.add_argument("--out", default=DEFAULT_DATASET)

    syn = sub.add_parser("synthetic", help="dataset synthétique (sans clé, sans edge)")
    syn.add_argument("--start", type=_date, required=True)
    syn.add_argument("--end", type=_date, required=True)
    syn.add_argument("--symbols")
    syn.add_argument("--seed", type=int, default=0)
    syn.add_argument("--out", default="data/backtest/synthetic_30m.jsonl")

    run = sub.add_parser("run", help="lance le backtest")
    run.add_argument("--data", default=DEFAULT_DATASET)
    run.add_argument("--eval-start", type=_date, help="début de l'évaluation (défaut : 1 an après le début)")
    run.add_argument("--eval-end", type=_date)
    run.add_argument("--grid", action="append", default=None,
                     help="paramètre=valeurs, répétable (défaut : " + DEFAULT_GRID[0] + ")")
    run.add_argument("--variants", default="complet,sans_modeles")
    run.add_argument("--workers", type=int, help="processus en parallèle (défaut : nombre de CPU)")
    run.add_argument("--out", default="data/backtest/report.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overlays = args.overlay if args.overlay is not None else [str(DEFAULT_OVERLAY)]
    cfg = load_config(args.config, tuple(overlays))
    symbols = (args.symbols.split(",") if getattr(args, "symbols", None) else list(cfg.feed.symbols))

    if args.command == "download":
        from trading_engine.backtest.dataset import download_bars
        from trading_engine.data.alpaca_history import AlpacaHistoricalClient

        client = AlpacaHistoricalClient.from_env(data_feed=args.feed)
        summary = download_bars(client, symbols, args.start, args.end, resolve_path(args.out),
                                timeframe=args.timeframe)
        print(summary)
        return 0

    if args.command == "synthetic":
        from trading_engine.backtest.dataset import write_synthetic_dataset

        n = write_synthetic_dataset(resolve_path(args.out), symbols, args.start, args.end, seed=args.seed)
        print(f"{n} barres écrites dans {args.out}")
        return 0

    from trading_engine.backtest.runner import format_report, parse_grid, run_backtest, to_json

    grid = parse_grid(args.grid if args.grid is not None else DEFAULT_GRID)
    report = run_backtest(
        args.data, config=args.config, overlays=overlays, eval_start=args.eval_start, eval_end=args.eval_end,
        grid=grid, variants=[v.strip() for v in args.variants.split(",") if v.strip()], workers=args.workers,
    )
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = format_report(report)
    out.write_text(to_json(report), encoding="utf-8")
    out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nRapport : {Path(args.out)} (+ {Path(args.out).with_suffix('.md').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
