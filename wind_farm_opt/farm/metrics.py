"""派生指标的统一定义。

所有"两个布局之间"或"部分占整体"的差值/百分比指标都必须经由此处计算，
控制台摘要、JSON 结果与可视化图表共用同一套规则，避免各处自行相除、
在零分母场景下崩溃或写出无意义的无穷值。

统一约定（相对类指标，如相对提升、损失降幅）
------------------------------------------------
* 分母为正            —— 返回正常的百分比数值；
* 分母与分子都为零    —— 返回 0.0（双方皆无，记作无变化，而非 0/0）；
* 分母为零、新值为正  —— 返回 None（"从无到有"没有相对变化的数学定义）；
* 任一输入为 None     —— 返回 None（数据不可用）。

绝对差值不涉及除法：两侧可用时恒为有限实数，任一缺失即为 None。
None 是唯一的"不可用"哨兵，可直接被 ``json.dump`` 序列化为 ``null``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .aep import FarmResult


# 节点信息缺失（例如只评估了基线、未执行优化）
MISSING = None


def _usable(value: Optional[float]) -> bool:
    """数值是否可参与计算：非 None 且为有限实数。"""
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def absolute_change(
    new_value: Optional[float],
    baseline_value: Optional[float],
) -> Optional[float]:
    """绝对差值 ``new_value - baseline_value``。

    两侧均可用时恒为有限实数（含双方都为零的 0.0）；
    任一缺失或非有限时返回 None。
    """
    if not (_usable(new_value) and _usable(baseline_value)):
        return None
    # 加 0.0 消除可能出现的负号零（-0.0），保证零差值的序列化与显示唯一。
    return float(new_value) - float(baseline_value) + 0.0


def relative_change(
    new_value: Optional[float],
    baseline_value: Optional[float],
) -> Optional[float]:
    """相对变化百分比 ``(new - baseline) / baseline * 100``（带符号）。

    分母为正时正常计算；分母为零且双方都为零时返回 0.0；
    分母为零而新值为正（相对提升无定义）或任一数据不可用时返回 None。
    本函数只服务于物理上非负的指标（发电量、容量系数等），负分母同样视为不可用。
    """
    if not (_usable(new_value) and _usable(baseline_value)):
        return None

    baseline_value = float(baseline_value)
    new_value = float(new_value)

    if baseline_value > 0.0:
        return (new_value - baseline_value) / baseline_value * 100.0 + 0.0
    if baseline_value == 0.0 and new_value == 0.0:
        return 0.0
    return None


def relative_reduction(
    baseline_value: Optional[float],
    new_value: Optional[float],
) -> Optional[float]:
    """损失降幅百分比 ``(baseline - new) / baseline * 100``（带符号）。

    正值表示损失减小。判定规则与 :func:`relative_change` 完全一致：
    分母为零且双方都为零时返回 0.0；分母为零而新值为正或数据缺失时返回 None。
    """
    change = relative_change(new_value, baseline_value)
    if change is None:
        return None
    # 加 0.0：对零降幅取负会得到 -0.0，统一归一化为 +0.0。
    return -change + 0.0


def safe_percent(part: Optional[float], whole: Optional[float]) -> Optional[float]:
    """部分占整体的百分比 ``part / whole * 100``。

    整体为正时正常计算；整体为零（或非正）时返回 0.0，与历史约定一致
    （零发电量场地的尾流损失率、容量系数记为 0%）；数据不可用时返回 None。
    """
    if not (_usable(part) and _usable(whole)):
        return None

    whole = float(whole)
    part = float(part)

    if whole > 0.0:
        return part / whole * 100.0
    return 0.0


def format_percent(value: Optional[float], decimals: int = 2, signed: bool = True) -> str:
    """将相对类指标格式化为控制台文本，None 统一显示为 ``N/A``。"""
    if value is None:
        return "N/A"
    spec = f"{{:{'+' if signed else ''}.{decimals}f}}%"
    return spec.format(float(value))


def format_signed_number(
    value: Optional[float],
    decimals: int = 2,
    unit: str = "",
) -> str:
    """将绝对差值格式化为带符号的控制台文本，None 统一显示为 ``N/A``。"""
    if value is None:
        return "N/A"
    text = f"{float(value):+.{decimals}f}"
    return f"{text}{unit}" if unit else text


@dataclass(frozen=True)
class LayoutComparison:
    """基线布局与优化后布局之间的派生对比指标。

    所有字段均可为 None（不可用），且全部为 JSON 原生可序列化类型。

    Parameters
    ----------
    aep_change_mwh : Optional[float]
        净AEP绝对差值（优化后 - 基线），MWh/年
    aep_change_pct : Optional[float]
        净AEP相对提升（%）；基线发电量为零而优化后为正时为 None
    wake_loss_reduction_pct : Optional[float]
        尾流损失率的相对降幅（%）；基线无尾流损失而优化后有时为 None
    """

    aep_change_mwh: Optional[float]
    aep_change_pct: Optional[float]
    wake_loss_reduction_pct: Optional[float]

    def to_dict(self) -> dict[str, Optional[float]]:
        """转换为 JSON 友好的字典（单位与历史 results.json 保持一致）。"""
        return {
            "aep_improvement_pct": self.aep_change_pct,
            "additional_aep_gwh": (
                self.aep_change_mwh / 1e3 if self.aep_change_mwh is not None else None
            ),
            "loss_reduction_pct": self.wake_loss_reduction_pct,
        }


def compare_farm_results(
    baseline_result: "Optional[FarmResult]",
    optimized_result: "Optional[FarmResult]",
) -> Optional[LayoutComparison]:
    """按统一规则计算两个布局结果之间的全部对比指标。

    任一结果缺失（数据不可用）时返回 None，由消费端序列化为 ``null``。
    """
    if baseline_result is None or optimized_result is None:
        return None

    aep_change_mwh = absolute_change(optimized_result.net_aep, baseline_result.net_aep)
    aep_change_pct = relative_change(optimized_result.net_aep, baseline_result.net_aep)
    wake_loss_reduction_pct = relative_reduction(
        baseline_result.wake_loss_pct,
        optimized_result.wake_loss_pct,
    )

    return LayoutComparison(
        aep_change_mwh=aep_change_mwh,
        aep_change_pct=aep_change_pct,
        wake_loss_reduction_pct=wake_loss_reduction_pct,
    )
