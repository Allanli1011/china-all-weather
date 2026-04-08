"""配置加载模块 — 从 YAML 文件读取并验证策略配置"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    raw: dict[str, Any]
    config_path: Path
    root_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root_dir = self.config_path.parent.parent

    # ── 子配置访问器 ───────────────────────────────────────────────────────

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw["universe"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def risk_parity(self) -> dict[str, Any]:
        return self.raw["risk_parity"]

    @property
    def rollover(self) -> dict[str, Any]:
        return self.raw["rollover"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]

    @property
    def output(self) -> dict[str, Any]:
        return self.raw["output"]

    # ── 常用快捷属性 ───────────────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_futures_dir(self) -> Path:
        return self.data_dir / "raw" / "futures"

    @property
    def raw_contracts_dir(self) -> Path:
        return self.data_dir / "raw" / "contracts"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / self.output["reports_dir"]

    @property
    def data_start_date(self) -> str:
        return self.data["start_date"]

    @property
    def initial_capital(self) -> float:
        return float(self.backtest["initial_capital"])

    @property
    def rebalance_threshold(self) -> float:
        return float(self.backtest["rebalance_threshold"])

    @property
    def benchmark_symbol(self) -> str:
        return self.backtest["benchmark_symbol"]


def load_settings(config_path: str | Path) -> Settings:
    """从 YAML 文件加载策略配置。

    Args:
        config_path: YAML 配置文件路径（绝对或相对路径均可）

    Returns:
        Settings 实例

    Raises:
        FileNotFoundError: 配置文件不存在
        KeyError: 必要配置项缺失
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate_config(raw)
    return Settings(raw=raw, config_path=path)


def _validate_config(raw: dict[str, Any]) -> None:
    required_sections = ["universe", "data", "risk_parity", "rollover", "backtest", "output"]
    for section in required_sections:
        if section not in raw:
            raise KeyError(f"配置文件缺少必要节: [{section}]")

    universe = raw["universe"]
    required_classes = ["equity", "bond", "gold", "industrial_metal", "ferrous", "energy", "agriculture"]
    for cls in required_classes:
        if cls not in universe:
            raise KeyError(f"universe 配置缺少资产类别: [{cls}]")
