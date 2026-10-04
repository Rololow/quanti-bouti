"""Backtest en ligne de commande.

    # 1. données (clés Alpaca dans l'environnement ; une seule fois)
    python -m trading_engine.backtest download --start 2018-01-01 --end 2026-09-30

    # 2. backtest : moteur (2 variantes x grille) + références + walk-forward
    python -m trading_engine.backtest run

    # sans clé : dataset synthétique pour vérifier la chaîne (aucun edge)
    python -m trading_engine.backtest synthetic --start 2022-01-01 --end 2024-12-31

    # historique quotidien long (CSV Stooq / Yahoo, un fichier par symbole)
    python -m trading_engine.backtest import-csv SPY=spy_us_d.csv QQQ=qqq_us_d.csv TLT=tlt_us_d.csv GLD=gld_us_d.csv
    python -m trading_engine.backtest --overlay config/backtest_daily.yaml run --data data/backtest/daily.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
import json
from datetime import date, timedelta
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

    dd = sub.add_parser("download-daily", help="historique quotidien gratuit (Yahoo ou Stooq) -> dataset 1d")
    dd.add_argument("--start", type=_date, default=date(2005, 1, 1))
    dd.add_argument("--end", type=_date, default=date.today())
    dd.add_argument("--symbols")
    dd.add_argument("--source", choices=["yahoo", "stooq"], default="yahoo")
    dd.add_argument("--no-fx", action="store_true", help="ne pas télécharger les taux BCE")
    dd.add_argument("--net-dividends", action="store_true",
                    help="réinvestir les dividendes nets d'impôts (profil fiscal et instruments de la config)")
    dd.add_argument("--out", default="data/backtest/daily.jsonl")

    imp = sub.add_parser("import-csv", help="CSV quotidiens (Stooq, Yahoo) -> dataset 1d")
    imp.add_argument("files", nargs="+", help="SYMBOLE=chemin.csv")
    imp.add_argument("--start", type=_date)
    imp.add_argument("--end", type=_date)
    imp.add_argument("--no-fx", action="store_true", help="ne pas télécharger les taux BCE")
    imp.add_argument("--net-dividends", action="store_true",
                     help="réinvestir les dividendes nets d'impôts (profil fiscal et instruments de la config)")
    imp.add_argument("--out", default="data/backtest/daily.jsonl")

    run = sub.add_parser("run", help="lance le backtest")
    run.add_argument("--data", default=DEFAULT_DATASET)
    run.add_argument("--eval-start", type=_date, help="début de l'évaluation (défaut : 1 an après le début)")
    run.add_argument("--eval-end", type=_date)
    run.add_argument("--grid", action="append", default=None,
                     help="paramètre=valeurs, répétable (défaut : " + DEFAULT_GRID[0] + ")")
    run.add_argument("--variants", default="complet,sans_modeles,seance")
    run.add_argument("--cost-stress", type=float, default=2.0,
                     help="multiplicateur des coûts pour le stress (0 = désactivé)")
    run.add_argument("--bootstrap", type=int, default=2000, help="tirages du bootstrap")
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

    if args.command in ("import-csv", "download-daily"):
        from trading_engine.backtest.dataset import fx_path, import_daily_csv
        from trading_engine.tax.fx import fetch_ecb_rates

        out = resolve_path(args.out)
        if args.command == "download-daily":
            from trading_engine.backtest.sources import download_csvs

            files = download_csvs(symbols, args.start, args.end, out.parent / f"csv_{args.source}",
                                  source=args.source)
        else:
            files = {}
            for item in args.files:
                sym, _, file = item.partition("=")
                if not file:
                    parser.error(f"expected SYMBOLE=chemin.csv, got {item!r}")
                files[sym.strip().upper()] = file
        keep = None
        if args.net_dividends:
            from trading_engine.tax.profile import load_tax_profile
            from trading_engine.tax.tax_model import TaxModel

            if not cfg.tax.profile:
                parser.error("--net-dividends requires tax.profile in the config")
            model = TaxModel(load_tax_profile(resolve_path(cfg.tax.profile)), cfg.instruments)
            keep = {sym: model.distribution_keep(sym) for sym in files}
        summary = import_daily_csv(files, out, start=args.start, end=args.end, keep=keep)
        if not summary["bars"]:
            print(summary)
            return 1
        if not args.no_fx and summary["bars"]:
            try:
                rates = fetch_ecb_rates("USD", date.fromisoformat(summary["start"]) - timedelta(days=10),
                                        date.fromisoformat(summary["end"]))
                fx_path(out).write_text(json.dumps(rates.to_payload()), encoding="utf-8")
                summary["fx"] = len(rates)
            except Exception as exc:             # le backtest utilisera le taux fixe
                summary["fx"] = f"non téléchargé ({exc})"
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
        cost_stress=args.cost_stress, bootstrap_samples=args.bootstrap,
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
