"""Synthetic regression contracts; no provider data, credentials, or network."""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test_env"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model = load("portfolio_model", "4_modeling/production_betting_model.py")
matching = load("portfolio_matching", "test_env/player_matching.py")
tracker = load("portfolio_tracker", "test_env/model_recommendation_tracker.py")
parser = load("portfolio_parser", "1_raw_data_extracts/grass_parser.py")
mapper = load("portfolio_mapper", "2_raw_data_mapping/results_grass_mapping.py")


def player(name, identifier):
    return {"player_name": name, "dg_id": identifier,
            "first_last": matching.flip_name_to_first_last(name)}


class OfflineContracts(unittest.TestCase):
    def setUp(self):
        blocker = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_softmax_stays_finite_for_large_scores(self):
        probabilities = model.softmax_probabilities([10000, 9999, 9998])
        self.assertTrue(all(math.isfinite(p) for p in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_negative_edge_allocates_nothing(self):
        self.assertEqual(model.calculate_kelly_stake(0.2, 2, 0.25, 1000), (0, 0))

    def test_allocation_respects_budget_cap(self):
        fraction, amount = model.calculate_kelly_stake(0.95, 3, 0.5, 1000, max_pct=0.02)
        self.assertEqual(fraction, 0.02)
        self.assertEqual(amount, 20)

    def test_odds_conversion(self):
        self.assertEqual(model.parse_american_odds("+150"), 2.5)
        self.assertEqual(model.parse_american_odds("-200"), 1.5)
        self.assertIsNone(model.parse_american_odds("invalid"))

    def test_normalization_preserves_identity_across_formats(self):
        self.assertEqual(matching.normalize_name("Démø, Ávery"), matching.normalize_name("Ávery Démø"))
        lookup = {"avery demo": player("Demo, Avery", 1)}
        result = matching.match_player_to_field("Ávery Demo", field_lookup=lookup)
        self.assertEqual(result["canonical_name"], "Avery Demo")
        self.assertEqual(result["match_tier"], 1)

    def test_ambiguous_initial_abstains(self):
        lookup = {"avery demo": player("Demo, Avery", 1), "alex demo": player("Demo, Alex", 2)}
        with contextlib.redirect_stderr(io.StringIO()):
            result = matching.match_player_to_field("A. Demo", field_lookup=lookup)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["datagolf_name"])

    def test_snapshot_deduplicates_reordered_recommendations(self):
        rows = [{"player_name": "Avery Demo", "market": "win", "stake": 10},
                {"player_name": "Blake Sample", "market": "win", "stake": 5}]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tracker, "MODEL_SNAPSHOTS_DIR", Path(directory)), contextlib.redirect_stdout(io.StringIO()):
                first = tracker.save_model_snapshot({"recommendations": rows}, tournament_name="Example Open")
                repeat = tracker.save_model_snapshot({"recommendations": rows[::-1]}, tournament_name="Example Open")
            self.assertIsNotNone(first)
            self.assertIsNone(repeat)
            self.assertEqual(json.loads(first.read_text())["recommendations"], rows)

    def test_api_helpers_refuse_missing_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            for reader in [parser.read_api_key, mapper.load_api_key]:
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                    reader()

    def test_provider_urls_reject_local_files_credentials_and_other_hosts(self):
        for url in ['file:///etc/passwd', 'http://www.gcsaa.org/example.pdf',
                    'https://example.com/example.pdf', 'https://gcsaa.org.example.com/a.pdf',
                    'https://user:example@' + 'gcsaa.org/a.pdf', 'https://localhost/a.pdf']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                parser.validate_source_url(url)
        self.assertEqual(parser.validate_source_url('https://www.gcsaa.org/example.pdf'),
                         'https://www.gcsaa.org/example.pdf')

    def test_provider_redirects_validate_destination(self):
        with self.assertRaises(ValueError):
            parser.ProviderRedirectHandler().redirect_request(
                None, None, 302, 'redirect', {}, 'https://example.com/unreviewed.pdf')

    def test_api_failure_does_not_include_response_details(self):
        def fail(**kwargs):
            raise RuntimeError('fictional-response-content-that-must-stay-private')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        fake_sdk = SimpleNamespace(OpenAI=lambda **kwargs: client)
        with patch.dict(sys.modules, {'openai': fake_sdk}):
            for module in [parser, mapper]:
                with patch.object(module.time, 'sleep'):
                    with self.assertRaises(RuntimeError) as error:
                        module.call_openai('fictional prompt', 'unused', '', 1)
                    self.assertNotIn('fictional-response-content', str(error.exception))


if __name__ == "__main__":
    unittest.main()
