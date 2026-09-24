"""Prompt-cache repayment gate for compaction (``agent/compression_economics.py``).

The gate is opt-in and fail-open, so the tests that matter are the ones that pin those two
properties: a disabled or malformed configuration must reproduce historical behaviour exactly, and
no input may ever produce "defer" when window pressure says compaction is due.
"""

import pytest

from types import SimpleNamespace
from typing import Any

from agent.compression_economics import (
    DEFAULT_CACHE_WRITE_READ_RATIO,
    DEFAULT_PAYBACK_HORIZON_REQUESTS,
    PRESSURE_OVERRIDE_FRACTION,
    cache_rewrite_penalty_tokens,
    evaluate_compaction_economics,
    payback_requests,
    resolve_cache_write_read_ratio,
    resolve_payback_horizon_requests,
)


class TestRatioResolution:
    @pytest.mark.parametrize(
        "raw",
        [None, True, False, "garbage", [], {}, float("nan"), float("inf"), float("-inf"), -1.0, -0.5],
    )
    def test_unusable_values_disable_the_gate(self, raw):
        """The gate is opt-in: a value we cannot model must not enable a cost model."""
        assert resolve_cache_write_read_ratio(raw) == 0.0

    @pytest.mark.parametrize("raw,expected", [(12.5, 12.5), ("12.5", 12.5), (1.0, 1.0), (3, 3.0)])
    def test_usable_values_pass_through(self, raw, expected):
        assert resolve_cache_write_read_ratio(raw) == expected

    def test_default_matches_the_published_anthropic_style_multiplier(self):
        assert DEFAULT_CACHE_WRITE_READ_RATIO == 12.5


class TestHorizonResolution:
    @pytest.mark.parametrize("raw", [None, True, "garbage", [], {}])
    def test_unusable_values_fall_back_to_the_default(self, raw):
        assert resolve_payback_horizon_requests(raw) == DEFAULT_PAYBACK_HORIZON_REQUESTS

    @pytest.mark.parametrize("raw,expected", [(0, 0), (5, 5), ("7", 7), (-3, 0)])
    def test_usable_values_are_clamped_non_negative(self, raw, expected):
        assert resolve_payback_horizon_requests(raw) == expected


class TestPaybackArithmetic:
    def test_penalty_is_the_extra_cost_of_a_write_over_a_read(self):
        # 40K rewritten at 12.5x read price = 11.5 * 40K extra read-equivalent tokens.
        assert cache_rewrite_penalty_tokens(40_000, 12.5) == int(40_000 * 11.5)

    @pytest.mark.parametrize("ratio", [1.0, 0.5, 0.0, -2.0])
    def test_a_write_no_more_expensive_than_a_read_carries_no_penalty(self, ratio):
        assert cache_rewrite_penalty_tokens(40_000, ratio) == 0

    def test_payback_scales_inversely_with_reclaim(self):
        trivial = payback_requests(5_000, 200_000, 12.5)
        healthy = payback_requests(200_000, 200_000, 12.5)
        assert trivial > healthy > 0

    def test_a_rewrite_with_nothing_reclaimed_never_repays(self):
        assert payback_requests(0, 200_000, 12.5) == float("inf")

    def test_nothing_to_repay_is_zero_requests(self):
        assert payback_requests(10_000, 0, 12.5) == 0.0


class TestEvaluateCompactionEconomics:
    def test_disabled_ratio_never_defers(self):
        defer, reason = evaluate_compaction_economics(
            reclaimed_tokens=1, new_prefix_tokens=500_000, ratio=0.0, horizon_requests=3,
        )
        assert defer is False and reason == ""

    def test_window_pressure_outranks_economics(self):
        """Above the override fraction, compaction proceeds no matter how bad the payback."""
        defer, reason = evaluate_compaction_economics(
            reclaimed_tokens=1, new_prefix_tokens=500_000, ratio=12.5, horizon_requests=3,
            context_pressure=PRESSURE_OVERRIDE_FRACTION,
        )
        assert defer is False and reason == ""

    def test_a_trivial_reclaim_is_deferred(self):
        # Rewriting a 200K prefix to reclaim 5K needs ~460 requests to repay: pure waste.
        defer, reason = evaluate_compaction_economics(
            reclaimed_tokens=5_000, new_prefix_tokens=200_000, ratio=12.5, horizon_requests=3,
        )
        assert defer is True
        assert reason.startswith("cache-economics:")

    def test_a_healthy_reclaim_proceeds(self):
        # Reclaiming the whole prefix repays the rewrite in ~11 requests... which is still over a
        # horizon of 3, so the caller must raise the horizon to accept it.
        defer, _ = evaluate_compaction_economics(
            reclaimed_tokens=200_000, new_prefix_tokens=200_000, ratio=12.5, horizon_requests=12,
        )
        assert defer is False

    def test_horizon_zero_only_accepts_an_immediate_repayment(self):
        defer, _ = evaluate_compaction_economics(
            reclaimed_tokens=200_000, new_prefix_tokens=200_000, ratio=12.5, horizon_requests=0,
        )
        assert defer is True

    def test_no_reclaim_reports_a_distinct_reason(self):
        defer, reason = evaluate_compaction_economics(
            reclaimed_tokens=0, new_prefix_tokens=200_000, ratio=12.5, horizon_requests=3,
        )
        assert defer is True and reason == "cache-economics:no-reclaim"

    @pytest.mark.parametrize("pressure", [None, float("nan"), float("inf"), "garbage"])
    def test_unreadable_pressure_is_treated_as_unknown_not_as_pressure(self, pressure):
        """A bad window reading must not silently disable the gate (it stays economic-only)."""
        defer, _ = evaluate_compaction_economics(
            reclaimed_tokens=5_000, new_prefix_tokens=200_000, ratio=12.5, horizon_requests=3,
            context_pressure=pressure,
        )
        assert defer is True

    def test_a_bad_horizon_falls_back_instead_of_raising(self):
        defer, _ = evaluate_compaction_economics(
            reclaimed_tokens=5_000, new_prefix_tokens=200_000, ratio=12.5,
            horizon_requests="garbage",  # type: ignore[arg-type] - unvalidated config value
        )
        assert defer is True


class TestCompressorAdoptsTheGate:
    """The compressor reads the configured values and defaults to the gate being off."""

    @staticmethod
    def _compressor(**overrides: Any):
        from agent.context_compressor import ContextCompressor

        return ContextCompressor(
            model="test-model", threshold_percent=0.5, protect_first_n=3, protect_last_n=20,
            summary_target_ratio=0.2, quiet_mode=True, config_context_length=200_000,
            **overrides,
        )

    def test_gate_is_disabled_by_default(self):
        assert self._compressor().cache_write_read_ratio == 0.0

    def test_configured_values_are_adopted(self):
        cc = self._compressor(cache_write_read_ratio=12.5, cache_payback_horizon_requests=5)
        assert cc.cache_write_read_ratio == 12.5
        assert cc.cache_payback_horizon_requests == 5

    def test_a_malformed_ratio_degrades_to_disabled(self):
        cc = self._compressor(cache_write_read_ratio="not-a-number")
        assert cc.cache_write_read_ratio == 0.0

    def test_verdict_defers_when_the_reclaim_cannot_repay_the_rewrite(self):
        cc = self._compressor(cache_write_read_ratio=12.5, cache_payback_horizon_requests=3)
        # Force a compressible window and a prefix far larger than the reclaim. 150K of a 200K
        # window = 75%, below the pressure override, so the economic gate is the one deciding.
        cc._compress_window = lambda messages: (1, 2)  # noqa: SLF001 - deliberate seam
        messages = [{"role": "user", "content": "x" * 400} for _ in range(4)]
        defer, reason = cc._compaction_economics_verdict(messages, 150_000, False)  # noqa: SLF001
        assert defer is True and reason.startswith("cache-economics:")

    def test_a_manual_compress_never_defers(self):
        """`force=True` reports full pressure, so an explicit /compress always runs."""
        cc = self._compressor(cache_write_read_ratio=12.5, cache_payback_horizon_requests=3)
        cc._compress_window = lambda messages: (1, 2)  # noqa: SLF001 - deliberate seam
        messages = [{"role": "user", "content": "x" * 400} for _ in range(4)]
        defer, reason = cc._compaction_economics_verdict(messages, 150_000, True)  # noqa: SLF001
        assert defer is False and reason == ""

    def test_window_pressure_overrides_a_bad_payback(self):
        cc = self._compressor(cache_write_read_ratio=12.5, cache_payback_horizon_requests=3)
        cc._compress_window = lambda messages: (1, 2)  # noqa: SLF001 - deliberate seam
        messages = [{"role": "user", "content": "x" * 400} for _ in range(4)]
        # 190K of a 200K window = 95% > the override fraction.
        defer, _ = cc._compaction_economics_verdict(messages, 190_000, False)  # noqa: SLF001
        assert defer is False

    def test_verdict_fails_open_when_measurement_raises(self):
        cc = self._compressor(cache_write_read_ratio=12.5)

        def _boom(messages):
            raise RuntimeError("no window for you")

        cc._compress_window = _boom  # noqa: SLF001 - deliberate seam
        defer, reason = cc._compaction_economics_verdict([], 150_000, False)  # noqa: SLF001
        assert defer is False and reason == ""

    def test_verdict_survives_a_compressor_built_without_init(self):
        """A compressor built via ``__new__`` (tests, restored session state) must not raise its way
        to a skipped compaction: the gate resolves its own settings defensively."""
        from agent.context_compressor import ContextCompressor

        cc = ContextCompressor.__new__(ContextCompressor)
        assert cc._compaction_economics_verdict([], 150_000, False) == (False, "")  # noqa: SLF001


class TestConfigRegistration:
    """Every DEFAULT_CONFIG key needs a reader, and every reader a registration (hermes_cli/AGENTS.md)."""

    def test_defaults_are_registered_and_match_the_module(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        compression = DEFAULT_CONFIG["compression"]
        assert "cache_write_read_ratio" in compression
        assert "cache_payback_horizon_requests" in compression
        # Registered default must equal what the module treats as "off"/"default".
        assert resolve_cache_write_read_ratio(compression["cache_write_read_ratio"]) == 0.0
        assert resolve_payback_horizon_requests(
            compression["cache_payback_horizon_requests"]
        ) == DEFAULT_PAYBACK_HORIZON_REQUESTS

    def test_the_gate_reaches_a_gateway_cached_agent_invalidation_list(self):
        """A baked-in setting must invalidate the cached agent, or a mid-gateway edit is ignored."""
        from gateway.run import GatewayRunner

        keys = GatewayRunner._CACHE_BUSTING_CONFIG_KEYS  # noqa: SLF001 - the registration surface
        assert ("compression", "cache_write_read_ratio") in keys
        assert ("compression", "cache_payback_horizon_requests") in keys

    def test_a_temp_config_yaml_reaches_the_compressor_settings(self, tmp_path, monkeypatch):
        """The behaviour the consuming surface sees, through the real loader and parse path."""
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "compression:\n"
            "  cache_write_read_ratio: 12.5\n"
            "  cache_payback_horizon_requests: 7\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        from agent.agent_init import _parse_compression_config
        from hermes_cli.config import load_config

        settings = _parse_compression_config(SimpleNamespace(), load_config())
        assert settings.cache_write_read_ratio == 12.5
        assert settings.cache_payback_horizon_requests == 7

    def test_an_absent_key_parses_to_disabled(self):
        from agent.agent_init import _parse_compression_config

        settings = _parse_compression_config(SimpleNamespace(), {})
        assert settings.cache_write_read_ratio == 0.0
        assert settings.cache_payback_horizon_requests == DEFAULT_PAYBACK_HORIZON_REQUESTS
