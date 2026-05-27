# core/__init__.py - ForgeX v2.1

# ★★★ torchvision 循环导入终极修复 ★★★
#
# 完整故障链:
#   from transformers import LlamaForCausalLM
#     → modeling_llama → modeling_layers → processing_utils → image_utils
#       → image_utils.py L53: from torchvision.transforms import InterpolationMode
#         → Python 加载真实 torchvision/__init__.py
#           → torchvision/__init__.py L10:
#               from torchvision import _meta_registrations, datasets, io, ...
#             → _meta_registrations.py L25: @register_meta("roi_align")
#               → _meta_registrations.py L18: torchvision.extension._has_ops()
#                 → torchvision.extension 还没初始化 → 循环导入
#
# 为什么 watchdog 方案失败:
#   watchdog.find_module("torchvision") 时 health=None → 返回 None
#   → Python 的 PathFinder 找到真实 torchvision → 执行 __init__.py
#   → 循环导入在 __init__.py 内部爆炸 → watchdog 没有第二次机会
#
# 终极方案: 在 core 包加载时主动测试 torchvision
#   成功 → 不干预
#   失败 → 注入 mock + 安装 Guardian 永久拦截真实 torchvision 加载
#
import sys as _sys
import types as _types
import enum as _enum


class _InterpolationMode(_enum.IntEnum):
    """torchvision.transforms.InterpolationMode 精确兼容实现"""
    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    NEAREST_EXACT = 6


def _build_torchvision_mock():
    """构建完整的 torchvision mock 模块树"""
    import importlib.machinery as _im

    def _make_mod(name, is_pkg=False):
        m = _types.ModuleType(name)
        m.__file__ = "<forgex_shim>"
        # ★ 关键: 设置 __spec__，否则 importlib.util.find_spec() 崩溃
        #   datasets/config.py L140: importlib.util.find_spec("torchvision")
        #   → 读取 sys.modules["torchvision"].__spec__ → None → ValueError
        m.__spec__ = _im.ModuleSpec(name, loader=None, origin="<forgex_shim>",
                                     is_package=is_pkg)
        if is_pkg:
            m.__path__ = []
            m.__package__ = name
        else:
            m.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        return m

    tv = _make_mod("torchvision", is_pkg=True)
    tv.__version__ = "0.0.0+forgex_shim"

    ext = _make_mod("torchvision.extension")
    ext._has_ops = lambda: False

    meta = _make_mod("torchvision._meta_registrations")

    transforms = _make_mod("torchvision.transforms", is_pkg=True)
    transforms.InterpolationMode = _InterpolationMode

    transforms_fn = _make_mod("torchvision.transforms.functional")
    transforms_fn.InterpolationMode = _InterpolationMode

    tv.extension = ext
    tv._meta_registrations = meta
    tv.transforms = transforms

    registry = {
        "torchvision": tv,
        "torchvision.extension": ext,
        "torchvision._meta_registrations": meta,
        "torchvision.transforms": transforms,
        "torchvision.transforms.functional": transforms_fn,
    }
    for sub in ("datasets", "io", "models", "ops", "utils"):
        m = _make_mod(f"torchvision.{sub}", is_pkg=True)
        setattr(tv, sub, m)
        registry[f"torchvision.{sub}"] = m

    return registry


def _clean_torchvision_modules():
    """清除 sys.modules 中所有 torchvision 相关模块"""
    for k in [k for k in list(_sys.modules.keys())
              if k == "torchvision" or k.startswith("torchvision.")]:
        del _sys.modules[k]


class _TorchvisionGuardian:
    """Meta path finder: 监控所有 torchvision import。

    - 真实 torchvision 健康时: 不干预，让 Python 正常加载
    - 真实 torchvision 损坏时: 拦截并返回 mock
    """
    _mock_mode = False  # True = 强制用 mock（真实 torchvision 已确认损坏）

    def find_module(self, fullname, path=None):
        if not (fullname == "torchvision" or fullname.startswith("torchvision.")):
            return None
        if not self._mock_mode:
            # 真实 torchvision 可能健康 → 检查是否需要介入
            existing = _sys.modules.get(fullname)
            if existing is not None and not getattr(existing, "__file__", "") == "<forgex_shim>":
                return None  # 真实模块已加载，不干预
            # 没在 sys.modules → 让 Python 尝试真实 import
            # （如果真实 import 失败，transformers 会报错，我们在 safe_loader 里兜底）
            return None
        self._ensure_available(fullname)
        return self

    def load_module(self, fullname):
        if fullname not in _sys.modules:
            self._ensure_available(fullname)
        return _sys.modules[fullname]

    def find_spec(self, fullname, path, target=None):
        """Python 3.4+ 首选入口"""
        if not (fullname == "torchvision" or fullname.startswith("torchvision.")):
            return None
        if not self._mock_mode:
            existing = _sys.modules.get(fullname)
            if existing is not None and not getattr(existing, "__file__", "") == "<forgex_shim>":
                return None  # 真实模块，不干预
            return None  # 让 Python 尝试正常 import
        self._ensure_available(fullname)
        import importlib.util
        spec = importlib.util.spec_from_loader(fullname, loader=self)
        return spec

    def create_module(self, spec):
        return _sys.modules.get(spec.name)

    def exec_module(self, module):
        pass  # mock 不需要执行

    def activate_mock_mode(self):
        """切换到 mock 模式: 清除真实 torchvision，注入 mock，永久拦截"""
        self._mock_mode = True
        _clean_torchvision_modules()
        _sys.modules.update(_build_torchvision_mock())

    def _ensure_available(self, fullname):
        """确保 fullname 在 sys.modules 中存在"""
        if fullname in _sys.modules:
            mod = _sys.modules[fullname]
            if getattr(mod, "__spec__", None) is None:
                import importlib.machinery as _im
                mod.__spec__ = _im.ModuleSpec(fullname, loader=None,
                                               origin="<forgex_shim>",
                                               is_package=bool(getattr(mod, "__path__", None)))
            return
        mock_registry = _build_torchvision_mock()
        for key, mod in mock_registry.items():
            if key not in _sys.modules:
                _sys.modules[key] = mod
        tv = _sys.modules.get("torchvision")
        if tv:
            for key in list(_sys.modules.keys()):
                if key.startswith("torchvision.") and key.count(".") == 1:
                    setattr(tv, key.split(".", 1)[1], _sys.modules[key])
        if fullname not in _sys.modules:
            import importlib.machinery as _im
            m = _types.ModuleType(fullname)
            m.__file__ = "<forgex_shim>"
            m.__path__ = []
            m.__spec__ = _im.ModuleSpec(fullname, loader=None,
                                         origin="<forgex_shim>", is_package=True)
            _sys.modules[fullname] = m


def _torchvision_init():
    """主动测试 torchvision，失败则注入 mock + 永久保护。"""

    # 总是安装 Guardian（即使 torchvision 健康也需要监控）
    _install_guardian()

    # 1. 如果已有健康的 torchvision → 完成
    tv = _sys.modules.get("torchvision")
    if tv is not None:
        try:
            _ = tv.extension._has_ops
            _ = tv.transforms.InterpolationMode
            return  # 健康
        except Exception:
            pass  # 僵尸，继续到步骤 3

    # 2. 主动测试: 能正常 import 吗？
    if tv is None:
        try:
            import torchvision
            _ = torchvision.transforms.InterpolationMode
            _ = torchvision.extension._has_ops
            return  # 成功，真实 torchvision 可用
        except Exception:
            pass  # 失败，继续到步骤 3

    # 3. 失败! 激活 mock 模式
    guardian = _get_guardian()
    if guardian:
        guardian.activate_mock_mode()
    else:
        _clean_torchvision_modules()
        _sys.modules.update(_build_torchvision_mock())

    print("[ForgeX] torchvision 不可用或循环导入，已注入兼容 shim")


def _install_guardian():
    """安装 Guardian（幂等）"""
    if not any(isinstance(f, _TorchvisionGuardian) for f in _sys.meta_path):
        _sys.meta_path.insert(0, _TorchvisionGuardian())


def _get_guardian():
    """获取已安装的 Guardian 实例"""
    for f in _sys.meta_path:
        if isinstance(f, _TorchvisionGuardian):
            return f
    return None


# ★ 立即执行 ★
_torchvision_init()


# 兼容层
try:
    from core.safe_loader import _patch_transformers_compat as _ptc  # noqa: F401
except Exception:
    pass

from core.config import config, DATASETS_DIR, LORAS_DIR, MODELS_CACHE_DIR, LOGS_DIR, PROJECT_DIR
from core.logger import log
from core.utils import (
    safe_json_load, safe_json_save, human_size, get_timestamp, random_name,
    run_subprocess, run_in_thread, estimate_vram_mb, guess_model_size,
    detect_gpu, auto_train_params, detect_target_modules,
)
from core.task_queue import task_queue, TaskStatus
