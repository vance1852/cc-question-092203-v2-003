"""共享派生指标（farm.metrics）的单元测试。

覆盖四种分母状态：分母为正、双方皆零、仅新值为正、数据不可用。
"""

import math
from types import SimpleNamespace

import pytest

from wind_farm_opt.farm.metrics import (
    LayoutComparison,
    absolute_change,
    compare_farm_results,
    format_percent,
    format_signed_number,
    relative_change,
    relative_reduction,
    safe_percent,
)


# ---------------------------------------------------------------- absolute_change


class TestAbsoluteChange:
    def test_positive_baseline(self):
        assert absolute_change(12.0, 10.0) == pytest.approx(2.0)

    def test_both_zero_is_zero(self):
        assert absolute_change(0.0, 0.0) == 0.0

    def test_zero_baseline_positive_new(self):
        # 绝对差值不涉及除法，"从无到有"仍是明确的有限值
        assert absolute_change(7.5, 0.0) == pytest.approx(7.5)

    @pytest.mark.parametrize(
        "new,baseline",
        [(None, 1.0), (1.0, None), (None, None)],
    )
    def test_missing_data(self, new, baseline):
        assert absolute_change(new, baseline) is None

    @pytest.mark.parametrize(
        "new,baseline",
        [(float("inf"), 1.0), (1.0, float("nan")), (float("-inf"), 0.0)],
    )
    def test_non_finite_treated_as_unavailable(self, new, baseline):
        assert absolute_change(new, baseline) is None

    def test_returns_plain_float(self):
        result = absolute_change(3, 1)
        assert isinstance(result, float)
        assert math.isfinite(result)


# ---------------------------------------------------------------- relative_change


class TestRelativeChange:
    def test_positive_baseline(self):
        assert relative_change(12.0, 10.0) == pytest.approx(20.0)

    def test_negative_change_kept_signed(self):
        assert relative_change(9.0, 10.0) == pytest.approx(-10.0)

    def test_both_zero_is_zero(self):
        assert relative_change(0.0, 0.0) == 0.0

    def test_zero_baseline_positive_new_is_undefined(self):
        # 0 -> 正值 没有相对变化的数学定义
        assert relative_change(5.0, 0.0) is None

    @pytest.mark.parametrize(
        "new,baseline",
        [(None, 1.0), (1.0, None), (None, None)],
    )
    def test_missing_data(self, new, baseline):
        assert relative_change(new, baseline) is None

    @pytest.mark.parametrize(
        "new,baseline",
        [(1.0, float("inf")), (float("nan"), 1.0), (1.0, -2.0)],
    )
    def test_invalid_inputs(self, new, baseline):
        assert relative_change(new, baseline) is None


# ------------------------------------------------------------- relative_reduction


class TestRelativeReduction:
    def test_positive_baseline_loss(self):
        # 损失率 8% -> 6%，降幅 25%
        assert relative_reduction(8.0, 6.0) == pytest.approx(25.0)

    def test_loss_increase_kept_signed(self):
        # 损失反而变大时为负值
        assert relative_reduction(6.0, 8.0) == pytest.approx(-33.3333333333)

    def test_no_wake_both_zero(self):
        # 单机 / 无尾流：两侧损失率皆为 0，降幅明确记为 0
        result = relative_reduction(0.0, 0.0)
        assert result == 0.0
        # 不得带负号零，否则 JSON 会出现无意义的 -0.0
        assert math.copysign(1.0, result) == 1.0

    def test_zero_reduction_is_positive_zero_even_from_negative_zero(self):
        assert str(relative_reduction(-0.0, -0.0)) == "0.0"

    def test_zero_baseline_positive_new_is_undefined(self):
        assert relative_reduction(0.0, 3.0) is None

    @pytest.mark.parametrize(
        "baseline,new",
        [(None, 1.0), (1.0, None), (None, None)],
    )
    def test_missing_data(self, baseline, new):
        assert relative_reduction(baseline, new) is None


# ------------------------------------------------------------------ safe_percent


class TestSafePercent:
    def test_positive_whole(self):
        assert safe_percent(2.0, 8.0) == pytest.approx(25.0)

    def test_zero_whole_is_zero(self):
        assert safe_percent(0.0, 0.0) == 0.0

    def test_non_positive_whole(self):
        assert safe_percent(1.0, -5.0) == 0.0

    @pytest.mark.parametrize("part,whole", [(None, 1.0), (1.0, None)])
    def test_missing_data(self, part, whole):
        assert safe_percent(part, whole) is None


# -------------------------------------------------------------------- formatters


class TestFormatters:
    def test_percent_na(self):
        assert format_percent(None) == "N/A"

    def test_percent_signed(self):
        assert format_percent(2.345, decimals=2) == "+2.35%"
        assert format_percent(-2.345, decimals=2) == "-2.35%"
        assert format_percent(0.0, decimals=2) == "+0.00%"

    def test_signed_number_na(self):
        assert format_signed_number(None) == "N/A"

    def test_signed_number_unit(self):
        assert format_signed_number(1.5, unit=" GWh/年") == "+1.50 GWh/年"
        assert format_signed_number(-1.5, unit=" GWh/年") == "-1.50 GWh/年"


# ------------------------------------------------------------- compare_farm_results


def _farm(net_aep, wake_loss_pct):
    """compare_farm_results 只读取这两个字段，用 SimpleNamespace 充当 FarmResult。"""
    return SimpleNamespace(net_aep=net_aep, wake_loss_pct=wake_loss_pct)


class TestCompareFarmResults:
    def test_normal_layout_matches_direct_formula(self):
        # 普通多机布局：与直接列式计算在数值上严格一致
        baseline = _farm(10000.0, 8.0)
        optimized = _farm(10500.0, 6.0)
        cmp = compare_farm_results(baseline, optimized)

        assert isinstance(cmp, LayoutComparison)
        assert cmp.aep_change_mwh == pytest.approx(500.0)
        assert cmp.aep_change_pct == pytest.approx(5.0)
        assert cmp.wake_loss_reduction_pct == pytest.approx(25.0)

    def test_single_turbine_both_zero_loss(self):
        # 单机：无尾流损失，AEP 相同
        cmp = compare_farm_results(_farm(6720.0, 0.0), _farm(6720.0, 0.0))
        assert cmp.aep_change_mwh == 0.0
        assert cmp.aep_change_pct == 0.0
        assert cmp.wake_loss_reduction_pct == 0.0

    def test_no_wind_both_zero_aep(self):
        # 无风：两侧发电量与损失率皆零
        cmp = compare_farm_results(_farm(0.0, 0.0), _farm(0.0, 0.0))
        assert cmp.aep_change_mwh == 0.0
        assert cmp.aep_change_pct == 0.0
        assert cmp.wake_loss_reduction_pct == 0.0

    def test_zero_to_positive_aep_is_undefined(self):
        # 基线零发电量、优化后有发电量：相对类指标不可用，绝对差值仍有效
        cmp = compare_farm_results(_farm(0.0, 0.0), _farm(5000.0, 2.0))
        assert cmp.aep_change_mwh == pytest.approx(5000.0)
        assert cmp.aep_change_pct is None
        assert cmp.wake_loss_reduction_pct is None

    @pytest.mark.parametrize(
        "baseline,optimized",
        [(None, _farm(1.0, 1.0)), (_farm(1.0, 1.0), None), (None, None)],
    )
    def test_missing_result(self, baseline, optimized):
        assert compare_farm_results(baseline, optimized) is None

    def test_to_dict_is_json_serializable_and_stable_units(self):
        import json

        cmp = compare_farm_results(_farm(0.0, 0.0), _farm(5000.0, 2.0))
        data = cmp.to_dict()
        # additional_aep_gwh 以 GWh 给出（MWh/1e3）
        assert data["additional_aep_gwh"] == pytest.approx(5.0)
        assert data["aep_improvement_pct"] is None
        assert data["loss_reduction_pct"] is None

        # 全部状态（含 None）必须是严格 JSON，不允许 Infinity/NaN 标记
        strict = json.loads(
            json.dumps(data),
            parse_constant=lambda c: pytest.fail(f"非标准 JSON 标记: {c}"),
        )
        assert strict == data
