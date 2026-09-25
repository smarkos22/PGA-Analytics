"""Exercise existing model utilities using fictional data and no external services."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    model = load_module("portfolio_model", "4_modeling/production_betting_model.py")
    matching = load_module("portfolio_matching", "test_env/player_matching.py")
    players = [
        {"name": "Demo, Avery", "score": 0.6, "odds": "+250"},
        {"name": "Demo, Blake", "score": 0.2, "odds": "+200"},
        {"name": "Demo, Casey", "score": -0.1, "odds": "+150"},
    ]
    probabilities = model.softmax_probabilities([p["score"] for p in players])
    rows = []
    for player, probability in zip(players, probabilities):
        decimal_odds = model.parse_american_odds(player["odds"])
        fraction, stake = model.calculate_kelly_stake(
            probability, decimal_odds, kelly_fraction=0.125, bankroll=1000.0,
            max_pct=0.02, min_stake=5.0,
        )
        rows.append({
            "player": matching.normalize_name(player["name"]),
            "synthetic_probability": round(probability, 6),
            "decimal_odds": decimal_odds,
            "budget_fraction": fraction,
            "hypothetical_stake": stake,
        })
    print(json.dumps({"data": "fictional; not a forecast", "players": rows}, indent=2))


if __name__ == "__main__":
    main()
