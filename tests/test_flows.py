"""端到端流程测试：单机、无风、无尾流与普通多机布局。

每个场景都跑通"基线评估 -> 优化 -> 经济性 -> 图表 -> results.json"链路，
并校验控制台之外的两个消费端（JSON、对比图）与共享指标定义一致。
"""

import json
import math
import os

import numpy as np
import pytest

from wind_farm_opt.cli import WindFarmOptimizerCLI
from wind_farm_opt.config import (
    OptimizationConfig,
    VisualizationConfig,
    WindFarmConfig,
)
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import WindResource, WindSector
from wind_farm_opt.economy.costs import (
    EconomicAnalyzer,
    get_default_farm_cost,
    get_default_turbine_cost,
)
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.farm.metrics import compare_farm_results
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.visualization.plotting import plot_comparison, plot_convergence


def _strict_load(path):
    """读取 JSON，遇到 Infinity / -Infinity / NaN 标记即判失败。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(
            f,
            parse_constant=lambda c: pytest.fail(f"results.json 含非标准标记 {c}，不可序列化"),
        )


def _make_cli(tmp_path, n_turbines, *, population=6, iterations=2, heatmap=False):
    cfg = WindFarmConfig(
        n_turbines=n_turbines,
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=population,
            max_iterations=iterations,
            seed=42,
        ),
        visualization=VisualizationConfig(
            save_dir=str(tmp_path),
            save_plots=True,
            show_plots=False,
            plot_wake_heatmap=heatmap,
        ),
    )
    return WindFarmOptimizerCLI(cfg)


# ---------------------------------------------------------------- 单机示范项目


def test_single_turbine_full_flow(tmp_path):
    """单机项目以前在"尾流损失降幅"处 0/0 崩溃，现在必须全流程跑通。"""
    cli = _make_cli(tmp_path, 1)
    cli.run_full_analysis()

    data = _strict_load(tmp_path / "results.json")

    improvement = data["improvement"]
    # 单机无尾流：绝对差值 0、相对提升 0、损失降幅 0（皆为明确的 0.0，而非 null）
    assert improvement["additional_aep_gwh"] == pytest.approx(0.0)
    assert improvement["aep_improvement_pct"] == 0.0
    assert improvement["loss_reduction_pct"] == 0.0

    # 零降幅不得序列化成无意义的 -0.0
    raw = (tmp_path / "results.json").read_text(encoding="utf-8")
    assert '"loss_reduction_pct": -0.0' not in raw
    assert all(
        not isinstance(v, float) or math.copysign(1.0, v) == 1.0
        for v in improvement.values()
    )

    assert data["baseline"]["wake_loss_pct"] == 0.0
    assert data["optimized"]["wake_loss_pct"] == 0.0
    assert data["economic"]["lcoe_yuan_per_kwh"] > 0.0

    assert os.path.exists(tmp_path / "comparison.png")
    assert os.path.exists(tmp_path / "convergence.png")


# ---------------------------------------------------------------- 无风 / 零发电量


def _calm_wind_resource():
    """构造近无风资源：各扇区 Weibull 风速远低于切入风速，发电量恒为零。"""
    sectors = [
        WindSector(
            direction_center=15.0 + 30.0 * i,
            direction_width=30.0,
            frequency=1.0 / 12.0,
            mean_speed=0.1,
            weibull_k=2.1,
            weibull_c=0.113,
        )
        for i in range(12)
    ]
    return WindResource(sectors)


def test_no_wind_full_flow(tmp_path):
    """停机曲线（零基线发电量）下不得再出现除零或 Infinity。"""
    cli = _make_cli(tmp_path, 2)
    cli.wind_resource = _calm_wind_resource()
    cli.aep_calc = AEPCalculator(
        turbines=cli.turbines,
        wind_resource=cli.wind_resource,
        wake_model=cli.wake_model,
        wake_superposition=cli.config.superposition_method,
    )

    cli.run_full_analysis()

    assert cli.baseline_result.net_aep == 0.0
    assert cli.optimized_result.net_aep == 0.0

    data = _strict_load(tmp_path / "results.json")

    improvement = data["improvement"]
    assert improvement["additional_aep_gwh"] == 0.0
    assert improvement["aep_improvement_pct"] == 0.0
    assert improvement["loss_reduction_pct"] == 0.0

    # LCOE 在零发电量时无定义：序列化为 null，而不是 Infinity
    assert data["economic"]["lcoe_yuan_per_kwh"] is None
    assert data["economic"]["annual_revenue_wanyuan"] == 0.0
    assert data["economic"]["npv_yiyuan"] is not None
    assert np.isfinite(data["economic"]["npv_yiyuan"])

    assert os.path.exists(tmp_path / "comparison.png")


def test_lcoe_none_when_zero_energy():
    analyzer = EconomicAnalyzer(
        get_default_turbine_cost("V126-3.45MW"),
        get_default_farm_cost(),
        electricity_price=0.45,
    )
    assert analyzer.analyze(2, 3.45, 0.0).lcoe is None

    result = analyzer.analyze(2, 3.45, 20.0)
    assert result.lcoe is not None
    assert np.isfinite(result.lcoe)
    assert result.lcoe > 0.0


# ---------------------------------------------------------------- 无尾流布局


def _crosswind_wind_resource():
    """单扇区北风（风向 0°，风沿 -y 方向吹）。"""
    return WindResource([
        WindSector(
            direction_center=0.0,
            direction_width=360.0,
            frequency=1.0,
            mean_speed=8.5,
            weibull_k=2.1,
            weibull_c=9.6,
        )
    ])


def test_no_wake_perpendicular_layout(tmp_path):
    """两台风机垂直于风向布置，互不遮挡：尾流损失率为 0，对比指标明确为 0。"""
    from wind_farm_opt.core.turbine import create_default_turbine

    turbines = [create_default_turbine("V126-3.45MW") for _ in range(2)]
    wind_resource = _crosswind_wind_resource()
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=wind_resource,
        wake_model=JensenWake(0.07),
        speed_step=1.0,
    )

    # 沿 x 轴并排；风向沿 -y，沿风向投影为零，完全无尾流
    positions_a = np.array([[0.0, 0.0], [1000.0, 0.0]])
    positions_b = np.array([[0.0, 0.0], [1500.0, 0.0]])

    result_a = calc.compute_farm_aep(positions_a)
    result_b = calc.compute_farm_aep(positions_b)

    assert result_a.gross_aep > 0.0
    assert result_a.total_wake_loss == pytest.approx(0.0, abs=1e-9)
    assert result_a.wake_loss_pct == 0.0

    comparison = compare_farm_results(result_a, result_b)
    assert comparison.wake_loss_reduction_pct == 0.0
    assert comparison.aep_change_pct == 0.0

    # 对比图消费端必须接受零损失率布局
    plot_comparison(
        result_a,
        result_b,
        save_path=str(tmp_path / "nowake_comparison.png"),
        show=False,
    )
    assert os.path.exists(tmp_path / "nowake_comparison.png")


def test_zero_baseline_positive_optimized_plot_and_json(tmp_path):
    """仅优化后为正：相对指标为 null，但图表与 JSON 都必须正常产出。"""
    from types import SimpleNamespace

    baseline = SimpleNamespace(
        net_aep=0.0,
        wake_loss_pct=0.0,
        capacity_factor=0.0,
        turbine_results=[object(), object()],
    )
    optimized = SimpleNamespace(
        net_aep=5000.0,
        wake_loss_pct=2.0,
        capacity_factor=20.0,
        turbine_results=[object(), object()],
    )

    comparison = compare_farm_results(baseline, optimized)
    assert comparison.aep_change_mwh == pytest.approx(5000.0)
    assert comparison.aep_change_pct is None
    assert comparison.wake_loss_reduction_pct is None

    payload = json.dumps(comparison.to_dict())
    assert json.loads(payload)["aep_improvement_pct"] is None

    plot_comparison(
        baseline,
        optimized,
        save_path=str(tmp_path / "zero_to_positive.png"),
        show=False,
    )
    assert os.path.exists(tmp_path / "zero_to_positive.png")


def test_optimization_skipped_marks_improvement_unavailable(tmp_path):
    """未执行优化（数据不可用）时 improvement 整体为 null，且不出对比图。"""
    cli = _make_cli(tmp_path, 2, population=4, iterations=1)
    cli.run_full_analysis(
        run_baseline=True,
        run_opt=False,
        run_econ=False,
        run_sweep=False,
        run_viz=True,
        save=True,
    )

    data = _strict_load(tmp_path / "results.json")
    assert data["improvement"] is None
    assert not os.path.exists(tmp_path / "comparison.png")


def test_convergence_plot_handles_zero_baseline(tmp_path):
    """收敛曲线的相对提升同样走共享定义：零基线时不崩溃、不显示百分比。"""
    from types import SimpleNamespace

    opt_result = SimpleNamespace(
        best_positions=np.zeros((1, 2)),
        best_fitness=0.0,
        best_generation=0,
        convergence_history=[0.0, 0.0],
        mean_history=[0.0, 0.0],
    )
    plot_convergence(
        opt_result,
        baseline_aep=0.0,
        save_path=str(tmp_path / "conv_zero.png"),
        show=False,
    )
    assert os.path.exists(tmp_path / "conv_zero.png")


# ---------------------------------------------------------------- 普通多机布局精度


def test_normal_layout_preserves_existing_precision(tmp_path):
    """正常多机项目的派生指标必须与旧公式逐位一致（仅集中了定义，未改数值）。"""
    cli = _make_cli(tmp_path, 12, population=8, iterations=3)
    cli.run_full_analysis(run_econ=True, run_sweep=False)

    base = cli.baseline_result
    opt = cli.optimized_result
    assert base.net_aep > 0.0  # 普通布局前提成立

    expected_improvement = (opt.net_aep - base.net_aep) / base.net_aep * 100.0
    expected_additional_gwh = (opt.net_aep - base.net_aep) / 1e3
    expected_reduction = (
        (base.wake_loss_pct - opt.wake_loss_pct) / base.wake_loss_pct * 100.0
        if base.wake_loss_pct > 0.0
        else 0.0
    )

    data = _strict_load(tmp_path / "results.json")
    improvement = data["improvement"]

    assert improvement["aep_improvement_pct"] == pytest.approx(
        expected_improvement, rel=1e-12
    )
    assert improvement["additional_aep_gwh"] == pytest.approx(
        expected_additional_gwh, rel=1e-12
    )
    assert improvement["loss_reduction_pct"] == pytest.approx(
        expected_reduction, rel=1e-12, abs=1e-12
    )

    # 布局自身指标经 safe_percent 中转，数值与历史结果一致
    assert data["baseline"]["wake_loss_pct"] == pytest.approx(base.wake_loss_pct)
    assert data["optimized"]["wake_loss_pct"] == pytest.approx(opt.wake_loss_pct)
    assert np.isfinite(data["economic"]["lcoe_yuan_per_kwh"])
    assert os.path.exists(tmp_path / "comparison.png")
