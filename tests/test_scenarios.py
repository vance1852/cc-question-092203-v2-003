"""完整流程场景测试。

覆盖任务要求的四类项目：
1. 单机示范项目（尾流占比恒为 0，损失降幅不再除零）；
2. 无风项目（停机/零发电量，相对提升为 0%、LCOE 为 null 而非 Infinity）；
3. 无尾流项目（自定义零亏损尾流模型，损失占比为 0）；
4. 普通多机项目（回归既有数值精度）。

每个场景都走通：布局生成/优化 -> FarmResult -> 统一对比指标 ->
JSON 序列化(allow_nan=False) -> 对比图渲染。
"""

import json

import numpy as np
import pytest

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake, WakeModel
from wind_farm_opt.core.wind_resource import WindResource, WindSector, create_default_wind_resource
from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.farm.metrics import compare_layouts, finite_float
from wind_farm_opt.optimization.baseline import generate_grid_layout
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.economy.costs import EconomicAnalyzer, get_default_farm_cost, get_default_turbine_cost
from wind_farm_opt.visualization.plotting import plot_comparison, plot_convergence


# 变更前在普通多机场景（seed=42）记录的基准数值，用于精度回归。
NORMAL_SNAPSHOT = {
    "baseline_net": 61181.63117024197,
    "optimized_net": 62236.24410552086,
    "baseline_loss_pct": 24.262702087884318,
    "improvement_pct": 1.723741121488509,
    "loss_reduction_pct": 5.380748375372877,
    "additional_mwh": 1054.612935278892,
}


class NoWake(WakeModel):
    """无尾流模型：所有速度亏损恒为零（自由来流）。"""

    def velocity_deficit(self, distance, rotor_diameter, thrust_coefficient):
        arr = np.zeros_like(np.asarray(distance, dtype=np.float64), dtype=np.float64)
        return arr if arr.ndim > 0 else 0.0

    def wake_radius(self, distance, rotor_diameter):
        arr = np.full_like(np.asarray(distance, dtype=np.float64), 0.5 * rotor_diameter)
        return arr if arr.ndim > 0 else float(0.5 * rotor_diameter)

    def radial_profile(self, radial_dist, wake_radius):
        arr = np.zeros_like(np.asarray(radial_dist, dtype=np.float64), dtype=np.float64)
        return arr if arr.ndim > 0 else 0.0


def make_zero_wind_resource(n_sectors: int = 12) -> WindResource:
    """构造可计算的近似无风资源：威布尔尺度 c 极小，积分风速区间内功率恒为零。"""
    width = 360.0 / n_sectors
    c = 0.1
    k = 2.1
    sectors = [
        WindSector(
            direction_center=float(width / 2.0 + i * width),
            direction_width=float(width),
            frequency=1.0 / n_sectors,
            mean_speed=0.1,
            weibull_k=k,
            weibull_c=c,
        )
        for i in range(n_sectors)
    ]
    return WindResource(sectors)


def run_pipeline(turbines, wind_resource, wake_model, boundary, n_turb, seed=42,
                 pop_size=8, generations=3, speed_step=1.0):
    """执行与 CLI 等价的核心流程，返回基线/优化结果和优化器结果。"""
    diameters = np.array([t.rotor_diameter for t in turbines])
    rng = np.random.default_rng(seed)

    baseline_positions = generate_grid_layout(
        boundary=boundary,
        n_turbines=n_turb,
        rotor_diameters=diameters,
        min_multiple=5.0,
        rng=rng,
    )

    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=wind_resource,
        wake_model=wake_model,
        speed_step=speed_step,
    )

    baseline = calc.compute_farm_aep(baseline_positions)

    ga = GeneticAlgorithm(
        n_turbines=n_turb,
        rotor_diameters=diameters,
        boundary=boundary,
        fitness_fn=calc.evaluate_layout,
        config=GAConfig(
            population_size=pop_size,
            max_generations=generations,
            min_spacing_multiple=5.0,
            seed=seed,
        ),
    )
    optimize_result = ga.optimize(verbose=False)
    optimized = calc.compute_farm_aep(optimize_result.best_positions)

    return baseline_positions, baseline, optimize_result, optimized


def assert_serializable(payload, path):
    """以严格 JSON 写盘并回读，拒绝 NaN/Infinity token。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


# ---------------------------------------------------------------------------
# 1. 单机示范项目
# ---------------------------------------------------------------------------

def test_single_turbine_full_pipeline(tmp_path):
    turbines = [create_default_turbine("V126-3.45MW")]
    wr = create_default_wind_resource(num_sectors=12, dominant_direction=270.0, mean_speed=8.5)
    boundary = create_rectangular_boundary(3500, 3500)

    _, baseline, opt_meta, optimized = run_pipeline(
        turbines, wr, JensenWake(0.07),
        boundary, n_turb=1, pop_size=6, generations=3,
    )

    assert baseline.net_aep > 0
    assert baseline.wake_loss_pct == 0.0
    assert optimized.wake_loss_pct == 0.0

    comparison = compare_layouts(baseline, optimized)
    # 关键回归：单机零尾流下损失降幅必须是 0.0，而不是除零崩溃。
    assert comparison.loss_reduction_pct == 0.0
    assert comparison.aep_improvement_pct == 0.0
    assert comparison.additional_aep_mwh == pytest.approx(0.0, abs=1e-9)

    payload = {"improvement": comparison.__dict__}
    reloaded = assert_serializable(payload, tmp_path / "single.json")
    assert reloaded["improvement"]["loss_reduction_pct"] == 0.0

    plot_comparison(baseline, optimized, save_path=str(tmp_path / "single_cmp.png"))
    plot_convergence(opt_meta, baseline.net_aep, save_path=str(tmp_path / "single_conv.png"))
    assert (tmp_path / "single_cmp.png").exists()


# ---------------------------------------------------------------------------
# 2. 无风项目（发电量为零）
# ---------------------------------------------------------------------------

def test_zero_wind_full_pipeline(tmp_path):
    n_turb = 3
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    wr = make_zero_wind_resource()
    boundary = create_rectangular_boundary(3500, 3500)

    _, baseline, opt_meta, optimized = run_pipeline(
        turbines, wr, JensenWake(0.07),
        boundary, n_turb=n_turb, pop_size=6, generations=2,
    )

    assert baseline.gross_aep == 0.0
    assert baseline.net_aep == 0.0
    assert optimized.net_aep == 0.0

    comparison = compare_layouts(baseline, optimized)
    # 双方都为零：相对提升和损失降幅都明确为 0%。
    assert comparison.aep_improvement_pct == 0.0
    assert comparison.loss_reduction_pct == 0.0
    assert comparison.additional_aep_mwh == 0.0

    analyzer = EconomicAnalyzer(
        get_default_turbine_cost("V126-3.45MW"),
        get_default_farm_cost(),
        electricity_price=0.45,
    )
    econ = analyzer.analyze(
        n_turbines=n_turb,
        rated_power_per_turbine_MW=3.45,
        net_aep_GWh=optimized.net_aep / 1e3,
    )
    # 零发电量时 LCOE 为无穷大：共享归一化后写 null，JSON 不含 Infinity。
    assert not np.isfinite(econ.lcoe)
    payload = {
        "improvement": comparison.__dict__,
        "economic": {"lcoe_yuan_per_kwh": finite_float(econ.lcoe)},
    }
    reloaded = assert_serializable(payload, tmp_path / "zero_wind.json")
    assert reloaded["economic"]["lcoe_yuan_per_kwh"] is None

    plot_comparison(baseline, optimized, save_path=str(tmp_path / "zero_cmp.png"))
    assert (tmp_path / "zero_cmp.png").exists()


def test_zero_baseline_positive_optimized_metrics(tmp_path):
    """零基准但优化后有发电量：相对值为 null，绝对差值保留且可序列化。"""
    turbines = [create_default_turbine("V126-3.45MW")]
    wr = create_default_wind_resource(num_sectors=12)
    boundary = create_rectangular_boundary(3500, 3500)
    calc = AEPCalculator(turbines, wr, __import__("wind_farm_opt.core.wake", fromlist=["JensenWake"]).JensenWake(0.07))

    zero_wr = make_zero_wind_resource()
    calc_zero = AEPCalculator(turbines, zero_wr, __import__("wind_farm_opt.core.wake", fromlist=["JensenWake"]).JensenWake(0.07))
    pos = np.array([[0.0, 0.0]])

    baseline = calc_zero.compute_farm_aep(pos)
    optimized = calc.compute_farm_aep(pos)
    assert baseline.net_aep == 0.0 and optimized.net_aep > 0.0

    comparison = compare_layouts(baseline, optimized)
    assert comparison.aep_improvement_pct is None
    assert comparison.loss_reduction_pct == 0.0
    assert comparison.additional_aep_mwh == pytest.approx(optimized.net_aep)

    reloaded = assert_serializable({"improvement": comparison.__dict__}, tmp_path / "zero_base.json")
    assert reloaded["improvement"]["aep_improvement_pct"] is None


# ---------------------------------------------------------------------------
# 3. 无尾流多机项目
# ---------------------------------------------------------------------------

def test_no_wake_multi_turbine_pipeline(tmp_path):
    n_turb = 4
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    wr = create_default_wind_resource(num_sectors=12)
    boundary = create_rectangular_boundary(6000, 6000)

    _, baseline, opt_meta, optimized = run_pipeline(
        turbines, wr, NoWake(), boundary, n_turb=n_turb, pop_size=6, generations=2,
    )

    assert baseline.net_aep > 0
    assert baseline.total_wake_loss == 0.0
    assert baseline.wake_loss_pct == 0.0
    assert optimized.wake_loss_pct == 0.0

    comparison = compare_layouts(baseline, optimized)
    assert comparison.loss_reduction_pct == 0.0
    assert comparison.aep_improvement_pct == 0.0

    assert_serializable({"improvement": comparison.__dict__}, tmp_path / "no_wake.json")
    plot_comparison(baseline, optimized, save_path=str(tmp_path / "no_wake_cmp.png"))
    assert (tmp_path / "no_wake_cmp.png").exists()


# ---------------------------------------------------------------------------
# 4. 普通多机项目：数值精度回归
# ---------------------------------------------------------------------------

def test_normal_multi_turbine_snapshot(tmp_path):
    n_turb = 12
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    wr = create_default_wind_resource(num_sectors=12, dominant_direction=270.0, mean_speed=8.5)
    boundary = create_rectangular_boundary(3500, 3500)

    _, baseline, opt_meta, optimized = run_pipeline(
        turbines, wr, JensenWake(0.07),
        boundary, n_turb=n_turb, pop_size=10, generations=5,
    )

    assert baseline.net_aep == pytest.approx(NORMAL_SNAPSHOT["baseline_net"])
    assert optimized.net_aep == pytest.approx(NORMAL_SNAPSHOT["optimized_net"])
    assert baseline.wake_loss_pct == pytest.approx(NORMAL_SNAPSHOT["baseline_loss_pct"])

    comparison = compare_layouts(baseline, optimized)
    assert comparison.aep_improvement_pct == pytest.approx(NORMAL_SNAPSHOT["improvement_pct"])
    assert comparison.loss_reduction_pct == pytest.approx(NORMAL_SNAPSHOT["loss_reduction_pct"])
    assert comparison.additional_aep_mwh == pytest.approx(NORMAL_SNAPSHOT["additional_mwh"])

    reloaded = assert_serializable({"improvement": comparison.__dict__}, tmp_path / "normal.json")
    assert reloaded["improvement"]["aep_improvement_pct"] == pytest.approx(
        NORMAL_SNAPSHOT["improvement_pct"]
    )

    plot_comparison(baseline, optimized, save_path=str(tmp_path / "normal_cmp.png"))
    plot_convergence(opt_meta, baseline.net_aep, save_path=str(tmp_path / "normal_conv.png"))
    assert (tmp_path / "normal_cmp.png").exists()
