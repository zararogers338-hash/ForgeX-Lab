# core/config.py - ForgeX v2 統一配置系統
import os
import yaml
from pathlib import Path
from .paths import project_root

BASE_DIR = project_root()
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============================ 路徑系統 ============================
PROJECT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
LORAS_DIR = DATA_DIR / "loras"
MODELS_CACHE_DIR = DATA_DIR / "models_cache"
LOGS_DIR = DATA_DIR / "logs"
CONFIGS_DIR = DATA_DIR / "configs"

for d in [DATA_DIR, DATASETS_DIR, LORAS_DIR, MODELS_CACHE_DIR, LOGS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = Path(os.environ.get("FORGEX_CONFIG", str(CONFIGS_DIR / "forgex.yaml"))).expanduser()
# If user gives a relative path, resolve against project root
if not CONFIG_FILE.is_absolute():
    CONFIG_FILE = (PROJECT_DIR / CONFIG_FILE).resolve()


@dataclass
class TrainDefaults:
    rank: int = 64
    alpha: int = 128
    epochs: int = 3
    batch_size: int = 4
    lr: float = 2e-4
    max_seq_length: int = 2048
    q_lora: bool = True
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500


@dataclass
class ForgeXConfig:
    # 基礎設定
    server_host: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False

    # HuggingFace 設定
    hf_endpoint: str = "https://huggingface.co"
    hf_mirror: str = "https://hf-mirror.com"
    use_mirror: bool = False

    # 後端路徑（可自定義，不再硬編碼）
    oobabooga_dir: str = ""
    mergekit_path: str = "mergekit-yaml"

    # UI 默認選項（方便 WebUI 記住常用模型）
    default_base_model: str = ""

    # 默認訓練參數
    train_defaults: TrainDefaults = field(default_factory=TrainDefaults)

    # GPU 設定
    default_device: str = "auto"
    max_memory_mb: int = 0  # 0 = 自動


    def get(self, key: str, default=None):
        """dict-like access for legacy UI code with compatibility aliases."""
        aliases = {"port": "server_port", "host": "server_host"}
        real_key = aliases.get(key, key)
        return getattr(self, real_key, default)

    @classmethod
    def load(cls) -> "ForgeXConfig":
        """從 YAML 加載配置，不存在則創建默認"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # 處理嵌套 dataclass
                train_data = data.pop("train_defaults", {})
                cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
                if train_data:
                    cfg.train_defaults = TrainDefaults(**{k: v for k, v in train_data.items() if k in TrainDefaults.__dataclass_fields__})
                return cfg
            except Exception:
                pass
        cfg = cls()
        cfg.save()
        return cfg

    def save(self):
        """保存配置到 YAML"""
        data = asdict(self)
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @property
    def effective_hf_endpoint(self) -> str:
        return self.hf_mirror if self.use_mirror else self.hf_endpoint


# 全局單例
config = ForgeXConfig.load()

# 設置 HF 環境變量
if config.use_mirror:
    os.environ["HF_ENDPOINT"] = config.hf_mirror