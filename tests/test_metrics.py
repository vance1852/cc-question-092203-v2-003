"""共享派生指标的单元测试。

覆盖任务要求的四种情形：分母为正、双方都为零、仅优化后为正、
数据不可用（None/NaN/inf）。
"""

import math

import numpy as np
import pytest

from wind_farm_opt.farm.metrics import (
    LayoutComparison,
    absolute_change,
    compare_layouts,
    finite_float,
    format_percent,
    format_signed_quantity,
    relative_change,
)
from wind_farm_opt.farm.aep import FarmResult


# ---------------------------------------------------------------------------
# finite_float
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (1.5, 1.5),
        (3, 3.0),
        ("2.5", 2.5),
        (np.float64(1.25), 1.25),
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (-float("inf"), None),
        (True, None),
        ("abc", None),
    ],
)
def test_finite_float(value, expected):
    result = finite_float(value)
    if expected is None:
        assert result is None
    else:
        assert result == expected
        assert isinstance(result, float)


def test_finite_float_numpy_nan():
    assert finite_float(np.nan) is None
    assert finite_float(np.inf) is None


# ---------------------------------------------------------------------------
# absolute_change
# ---------------------------------------------------------------------------

def test_absolute_change_positive_denominator_regime():
    assert absolute_change(120.0, 100.0) == pytest.approx(20.0)


def test_absolute_change_both_zero():
    assert absolute_change(0.0, 0.0) == 0.0


def test_absolute_change_new_only_positive():
    # 仅优化后为正：绝对差值仍然有明确定义。
    assert absolute_change(50.0, 0.0) == 50.0


@pytest.mark.parametrize("new, baseline", [(None, 1.0), (1.0, None), (float("inf"), 1.0), (1.0, float("nan"))])
def test_absolute_change_unavailable(new, baseline):
    assert absolute_change(new, baseline) is None


# ---------------------------------------------------------------------------
# relative_change
# ---------------------------------------------------------------------------

def test_relative_change_denominator_positive():
    # 与原公式完全一致，保留既有数值精度。
    assert relative_change(120.0, 100.0) == pytest.approx(20.0)
    assert relative_change(95.0, 100.0) == pytest.approx(-5.0)


def test_relative_change_reduction_denominator_positive():
    assert relative_change(80.0, 100.0, reduction=True) == pytest.approx(20.0)
    assert relative_change(110.0, 100.0, reduction=True) == pytest.approx(-10.0)


def test_relative_change_both_zero():
    # 双方都为零：没有变化，明确定义为 0%。
    assert relative_change(0.0, 0.0) == 0.0
    assert relative_change(0.0, 0.0, reduction=True) == 0.0


def test_relative_change_new_only_positive():
    # 零基准、新值非正零基准、新值非零：有限百分比不存在，返回 None 而非 inf。
    assert relative_change(50.0, 0.0) is None
    assert relative_change(0.001, 0.0) is None
    assert relative_change(50.0, 0.0, reduction=True) is None


def test_relative_change_negative_baseline():
    assert relative_change(10.0, -5.0) is None


@pytest.mark.parametrize(
    "new, baseline",
    [
        (None, 1.0),
        (1.0, None),
        (float("nan"), 1.0),
        (1.0, float("inf")),
    ],
)
def test_relative_change_unavailable(new, baseline):
    assert relative_change(new, baseline) is None


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------

def test_format_percent():
    assert format_percent(1.7237) == "+1.72%"
    assert format_percent(-0.5) == "-0.50%"
    assert format_percent(0.0) == "+0.00%"
    assert format_percent(None) == "N/A"


def test_format_signed_quantity():
    assert format_signed_quantity(1.05, suffix=" GWh") == "+1.05 GWh"
    assert format_signed_quantity(None) == "N/A"


# ---------------------------------------------------------------------------
# compare_layouts
# ---------------------------------------------------------------------------

def _farm_result(net_aep: float, wake_loss_pct: float) -> FarmResult:
    return FarmResult(
        gross_aep=net_aep,
        net_aep=net_aep,
        total_wake_loss=0.0,
        wake_loss_pct=wake_loss_pct,
        capacity_factor=30.0,
        total_installed_capacity=3.45,
        turbine_results=[],
        sector_results={},
    )


def test_compare_layouts_missing_side():
    assert compare_layouts(None, _farm_result(1.0, 1.0)) is None
    assert compare_layouts(_farm_result(1.0, 1.0), None) is None
    assert compare_layouts(None, None) is None


def test_compare_layouts_normal():
    comparison = compare_layouts(_farm_result(100.0, 10.0), _farm_result(110.0, 8.0))
    assert isinstance(comparison, LayoutComparison)
    assert comparison.aep_improvement_pct == pytest.approx(10.0)
    assert comparison.additional_aep_mwh == pytest.approx(10.0)
    assert comparison.loss_reduction_pct == pytest.approx(20.0)


def test_compare_layouts_both_zero_is_serializable():
    comparison = compare_layouts(_farm_result(0.0, 0.0), _farm_result(0.0, 0.0))
    assert comparison.aep_improvement_pct == 0.0
    assert comparison.additional_aep_mwh == 0.0
    assert comparison.loss_reduction_pct == 0.0
    # 必须是 JSON 可序列化的有限值。
    import json

    payload = json.dumps(comparison.__dict__, allow_nan=False)
    assert json.loads(payload)["aep_improvement_pct"] == 0.0


def test_compare_layouts_optimized_only_positive():
    # 基线零发电量、优化后有发电量：相对值为 null，绝对差值保留。
    comparison = compare_layouts(_farm_result(0.0, 0.0), _farm_result(50.0, 0.0))
    assert comparison.aep_improvement_pct is None
    assert comparison.additional_aep_mwh == 50.0
    assert comparison.loss_reduction_pct == 0.0

    import json

    payload = json.dumps(comparison.__dict__, allow_nan=False)
    assert json.loads(payload)["aep_improvement_pct"] is None


def test_metrics_never_produce_inf_or_nan():
    grid = [0.0, 1.0, -1.0, float("nan"), float("inf"), None]
    for new in grid:
        for baseline in grid:
            assert absolute_change(new, baseline) in (None,) or math.isfinite(
                absolute_change(new, baseline)
            )
            rel = relative_change(new, baseline)
            assert rel is None or math.isfinite(rel)
