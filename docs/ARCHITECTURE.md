# ForgeX 架构说明 / Architecture

## 中文

ForgeX 采用“Gradio UI + 核心模块 + 本地数据目录”的结构。

### 入口层

- `run.bat` / `run.sh`：面向用户的一键启动脚本。
- `launcher.py`：创建 `.venv`、安装依赖、识别硬件并启动 UI。
- `main.py`：Gradio UI 主文件，组织数据、训练、编辑、导出等页面。

### 核心模块

- `core/dataset_manager.py`：数据集上传、预览、转换和清洗。
- `core/trainer.py`：SFT / DPO / LoRA / QLoRA 训练入口。
- `core/distiller.py`：本地或 API 教师蒸馏流程。
- `core/doc_curriculum.py`：文档解析、知识单元提取、课程式训练样本生成。
- `core/self_evolve.py`：自我进化训练流程。
- `core/multimodal.py`：多模态训练原型。
- `core/merger.py`：LoRA 合并与模型目录解析。
- `core/exporter.py`：GGUF 转换、量化和 Ollama 导出。
- `core/model_editor.py`：模型身份、提示词、知识库、安全规则和模型卡维护。
- `core/expansion.py` / `core/expander.py`：模型扩容、词表扩展、知识嫁接和 MoE upcycle。
- `core/simulation/`：训练显存、数据质量和任务估算。

### 数据目录

- `data/datasets/`：用户上传的数据集，默认不提交到 Git。
- `data/loras/`：训练输出，默认不提交到 Git。
- `data/models_cache/`：模型缓存，默认不提交到 Git。
- `data/configs/`：默认配置文件，可提交。

### 设计重点

- 本地优先：默认在本机运行，不依赖云端服务。
- 大文件隔离：模型、数据集、LoRA 输出不进入仓库。
- 可审计：开源版核心模块已还原为可读源码。
- 兼容优先：通过 `constraints.txt` 固定 Gradio/FastAPI/Starlette 兼容组合。

---

## English

ForgeX uses a “Gradio UI + core modules + local data directories” architecture.

### Entry Layer

- `run.bat` / `run.sh`: user-facing one-click launch scripts.
- `launcher.py`: creates `.venv`, installs dependencies, detects hardware, and launches UI.
- `main.py`: main Gradio UI file that organizes dataset, training, editing, and export pages.

### Core Modules

- `core/dataset_manager.py`: dataset upload, preview, conversion, and cleaning.
- `core/trainer.py`: SFT / DPO / LoRA / QLoRA training entry point.
- `core/distiller.py`: local or API teacher distillation workflow.
- `core/doc_curriculum.py`: document parsing, knowledge-unit extraction, and curriculum sample generation.
- `core/self_evolve.py`: self-evolution training workflow.
- `core/multimodal.py`: multimodal training prototype.
- `core/merger.py`: LoRA merge and model directory resolution.
- `core/exporter.py`: GGUF conversion, quantization, and Ollama export.
- `core/model_editor.py`: model identity, prompts, knowledge base, safety rules, and model cards.
- `core/expansion.py` / `core/expander.py`: model expansion, vocabulary expansion, knowledge grafting, and MoE upcycling.
- `core/simulation/`: VRAM, data quality, and training estimates.

### Data Directories

- `data/datasets/`: user datasets, ignored by Git.
- `data/loras/`: training outputs, ignored by Git.
- `data/models_cache/`: local model cache, ignored by Git.
- `data/configs/`: default config files, committed.

### Design Priorities

- Local-first: runs on the user machine by default.
- Large-file isolation: models, datasets, and LoRA outputs stay out of Git.
- Auditable: open-source build uses readable source code.
- Compatibility-first: `constraints.txt` pins a compatible Gradio/FastAPI/Starlette stack.
