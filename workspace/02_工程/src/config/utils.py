from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典；overlay 覆盖 base，子字典递归合并。"""
    result: Dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_module_overlay(base_config_path: str, module_id: str) -> Dict[str, Any]:
    """
    将 default.yaml 与指定模块 YAML 合并，不读取 active_module。
    用于并行挂载的子能力（如坐姿占位接口），与当前 active_module 无关。
    """
    base_path = Path(base_config_path).resolve()
    cfg = load_config(str(base_path))
    mod_file = base_path.parent / "modules" / f"{module_id}.yaml"
    if not mod_file.is_file():
        raise FileNotFoundError(f"未找到模块: {mod_file}")
    overlay = load_config(str(mod_file))
    return deep_merge(cfg, overlay)


def load_resolved_config(path: str) -> Dict[str, Any]:
    """
    读取 default.yaml，并按 active_module 合并 modules/<id>.yaml。
    新增业务能力请新增模块文件，无需改核心训练/推理代码。
    """
    base_path = Path(path).resolve()
    cfg = load_config(str(base_path))

    mod_id = cfg.get("active_module")
    if not mod_id:
        return cfg

    mod_file = base_path.parent / "modules" / f"{mod_id}.yaml"
    if not mod_file.is_file():
        raise FileNotFoundError(
            f"未找到任务模块配置: {mod_file}（active_module={mod_id}）"
        )

    overlay = load_config(str(mod_file))
    return deep_merge(cfg, overlay)
