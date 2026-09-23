"""布局对比的派生指标定义（单一事实来源）。

控制台摘要、JSON 结果和对比图都必须使用这里的函数，禁止在消费端
自行做除法，以免零分母崩溃或产生不可序列化的无穷值。

指标约定
--------
给定基线值 ``baseline`` 与优化后值 ``new``：

- 绝对差值 = ``new - baseline``，只要双方为有限数即可计算。
- 相对变化（百分数）：
    * ``baseline > 0``：正常比值；
    * ``baseline == 0 且 new == 0``：定义为 ``0.0``（双方均为零，
      没有变化）；
    * ``baseline == 0 且 new != 0``：定义为 ``None``——从零基准出发
      无法给出有限的相对百分比，绝对差值仍有意义；
    * ``baseline < 0``：业务指标不应为负，返回 ``None``。
- 任一输入缺失（``None``）或不是有限数（``NaN``/``inf``）：
  返回 ``None``，表示"数据不可用"。

所有数值结果均为 Python 原生 ``float``，可直接被 :mod:`json` 序列化；
不可用状态统一用 ``None``（JSON 中为 ``null``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .aep import FarmResult


def finite_float(value: object) -> Optional[float]:
    """把输入转换为有限的 Python float；不可用时返回 None。

    ``None``、布尔值、无法转换为浮点数的值以及 ``NaN``/``±inf``
    都视为数据不可用。这样得到的 None 可直接写入 JSON，
    避免出现 ``Infinity`` 这类非法 JSON token。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def absolute_change(new: object, baseline: object) -> Optional[float]:
    """绝对差值 ``new - baseline`` (MWh 等原始单位)。

    双方均为有限数时返回 Python float，否则返回 None。
    """
    new_value = finite_float(new)
    baseline_value = finite_float(baseline)
    if new_value is None or baseline_value is None:
        return None
    return new_value - baseline_value


def relative_change(
    new: object,
    baseline: object,
    reduction: bool = False,
) -> Optional[float]:
    """统一的相对变化（百分数）。

    Parameters
    ----------
    new:
        优化后指标值。
    baseline:
        基线指标值。
    reduction:
        为 ``True`` 时计算"损失降幅"
        ``(baseline - new) / baseline * 100``；
        否则计算"相对提升" ``(new - baseline) / baseline * 100``。

    Returns
    -------
    Optional[float]
        相对变化百分数；零基准下新值非零、负基准或数据不可用时为 None。
    """
    new_value = finite_float(new)
    baseline_value = finite_float(baseline)
    if new_value is None or baseline_value is None:
        return None

    if baseline_value > 0.0:
        change = baseline_value - new_value if reduction else new_value - baseline_value
        return change / baseline_value * 100.0

    if baseline_value == 0.0:
        # 双方都为零：没有变化，明确记为 0%；
        # 仅优化后为正：零基准无法定义有限百分比，记为不可用。
        return 0.0 if new_value == 0.0 else None

    # 负基准对发电量/损失类指标没有业务含义。
    return None


def format_percent(value: Optional[float], digits: int = 2) -> str:
    """把百分数指标格式化为控制台文本，不可用时显示 ``N/A``。"""
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}%"


def format_signed_quantity(
    value: Optional[float],
    suffix: str = "",
    digits: int = 2,
) -> str:
    """把带符号的绝对差值格式化为控制台文本，不可用时显示 ``N/A``。"""
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}{suffix}"


@dataclass(frozen=True)
class LayoutComparison:
    """两个布局（基线 vs 优化后）的派生对比指标。

    所有字段均可直接 JSON 序列化；不可用时为 None。

    Parameters
    ----------
    aep_improvement_pct : Optional[float]
        净 AEP 相对提升 (%)。
    additional_aep_mwh : Optional[float]
        额外净发电量 (MWh/year)，即绝对差值。
    loss_reduction_pct : Optional[float]
        尾流损失占比的相对降幅 (%)。
    """

    aep_improvement_pct: Optional[float]
    additional_aep_mwh: Optional[float]
    loss_reduction_pct: Optional[float]


def compare_layouts(
    baseline: Optional[FarmResult],
    optimized: Optional[FarmResult],
) -> Optional[LayoutComparison]:
    """根据两个 :class:`FarmResult` 计算统一的对比指标。

    任一侧结果缺失（流程未执行）时返回 None，由消费端决定如何呈现。
    """
    if baseline is None or optimized is None:
        return None

    return LayoutComparison(
        aep_improvement_pct=relative_change(optimized.net_aep, baseline.net_aep),
        additional_aep_mwh=absolute_change(optimized.net_aep, baseline.net_aep),
        loss_reduction_pct=relative_change(
            optimized.wake_loss_pct,
            baseline.wake_loss_pct,
            reduction=True,
        ),
    )
