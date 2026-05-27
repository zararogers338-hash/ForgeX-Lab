# ⚒️ ForgeX v3.2.0-open-source

**中文** | [English](#english)

ForgeX 是一个本地运行的 AI 模型训练、数据处理、蒸馏、模型编辑与部署工具箱。它把常见的大模型训练流程集中到一个 Gradio UI 里，适合个人研究、课程实验、模型微调、数据集清洗、LoRA 训练、GGUF/Ollama 导出与原型验证。

> 开源版默认不需要 `license.key`。原用户版中的轻量混淆模块已经还原为可读源码，便于审计、学习和二次开发。

## 这个项目是做什么的？

ForgeX 的目标是把“大模型训练工坊”做成本地可视化工具：你可以导入数据集，选择本地或 Hugging Face 模型，配置训练参数，运行 SFT/DPO/LoRA/QLoRA 训练，做 API 教师蒸馏，处理文档课程数据，尝试多模态训练，最后把模型合并、量化并导出到 GGUF 或 Ollama。

## 主要功能

- **数据集管理**：上传、预览、删除、格式转换、清洗、去重、质量分析。
- **标准训练**：SFT / DPO / LoRA / QLoRA，带智能参数推荐与显存估算。
- **API 教师蒸馏**：用 OpenAI 兼容接口或本地 Ollama 生成训练数据、批改答案、训练学生模型。
- **文档课程训练**：解析 TXT / Markdown / PDF / DOCX / JSONL，生成课程式训练样本。
- **自我进化训练**：从主题生成指令、生成候选回答、筛选偏好数据并迭代训练。
- **多模态训练**：面向图文/音文理解的 LLaVA-style 原型训练流程。
- **模型手术**：深度扩展、宽度扩展、词表扩展、知识嫁接、MoE upcycle。
- **模型合并与导出**：LoRA 合并、GGUF 转换、Ollama Modelfile 生成。
- **模型编辑器**：模型身份、系统提示词、知识库、安全护栏、推理参数与模型卡维护。
- **环境适配**：启动器自动创建 `.venv`，并根据 NVIDIA/CPU 环境安装 PyTorch。

## 快速开始

### 1. 准备环境

需要 Python **3.10+**。建议放在较短且可写的路径中，例如 Windows 下的 `C:\ForgeX`。

### 2. 启动

Windows：

```bat
run.bat
```

Linux / macOS：

```bash
chmod +x run.sh install.sh install_forgex.sh
./run.sh
```

首次运行会自动创建 `.venv` 并安装依赖。UI 默认监听：

```text
http://127.0.0.1:7860
```

### 3. 常用命令

```bash
python launcher.py run             # 安装依赖并启动 UI
python launcher.py install         # 只安装/更新运行环境
python launcher.py fingerprint     # 输出本机指纹（兼容旧授权机制）
python launcher.py license-info    # 显示旧 license 信息；开源版无需 license
python selfcheck.py                # 静态自检：Python 文件可编译、核心接口存在
python smoke_test.py               # 轻量冒烟测试
```

## 项目结构

```text
ForgeX/
├─ core/                    # 训练、蒸馏、导出、数据、模型编辑等核心模块
├─ core/simulation/         # 训练/显存估算与模拟工具
├─ data/configs/            # 默认配置
├─ data/datasets/           # 用户数据集目录（默认不入库）
├─ data/loras/              # 训练输出目录（默认不入库）
├─ data/models_cache/       # 模型缓存目录（默认不入库）
├─ docs/                    # 架构、路线图、开源说明
├─ examples/                # 示例数据
├─ main.py                  # Gradio UI 主入口
├─ launcher.py              # 跨平台启动器与环境安装器
├─ requirements.txt         # Python 依赖
└─ constraints.txt          # UI 兼容性约束
```

## 注意事项

- 本仓库**不包含模型权重**，也不包含原用户包中的大型训练数据文件。
- 训练大模型可能需要大量显存、磁盘空间和下载流量。
- 使用第三方模型、数据集、API 或论文资料时，请自行遵守其许可证、服务条款和版权要求。
- 部分模型加载流程会使用 `trust_remote_code=True`。只加载你信任的模型仓库。
- API Key 只应通过 UI 或环境变量临时提供，不要提交到 Git。

## 开源发布前检查清单

- [x] 核心混淆模块已还原为可读源码。
- [x] 删除 `license.key`。
- [x] 删除大型训练数据 `train_*.jsonl`。
- [x] 删除日志、临时文件和本地模型缓存。
- [x] 加入 `.gitignore`、`LICENSE`、贡献说明、安全说明、架构文档和示例数据。
- [x] `python -m py_compile` 静态编译通过。

## 许可证

本开源准备包默认使用 **MIT License**。如你希望改成 Apache-2.0、GPL-3.0 或保持闭源商业授权，请在发布前替换 `LICENSE`。

---

<a id="english"></a>

# ⚒️ ForgeX v3.2.0-open-source

ForgeX is a local AI model training, dataset processing, distillation, model editing, and deployment toolbox. It brings common LLM workflows into a Gradio-based UI for research, learning, fine-tuning, dataset cleaning, LoRA training, GGUF/Ollama export, and rapid prototyping.

> The open-source build does not require `license.key`. The lightweight packaging obfuscation from the user build has been converted back into readable source code for review, learning, and contribution.

## What is ForgeX?

ForgeX is designed as a local “LLM forge.” You can import datasets, choose local or Hugging Face models, configure training parameters, run SFT/DPO/LoRA/QLoRA jobs, perform API teacher distillation, build document-based curricula, experiment with multimodal training, and finally merge, quantize, or export models to GGUF/Ollama.

## Features

- **Dataset management**: upload, preview, delete, convert, clean, deduplicate, and analyze datasets.
- **Standard training**: SFT / DPO / LoRA / QLoRA with smart parameter recommendations and VRAM estimates.
- **API teacher distillation**: use OpenAI-compatible APIs or local Ollama to generate data, grade answers, and train student models.
- **Document curriculum training**: parse TXT / Markdown / PDF / DOCX / JSONL and generate curriculum-style training samples.
- **Self-evolution training**: generate instructions from topics, create candidates, build preference pairs, and iterate.
- **Multimodal training**: prototype LLaVA-style image-text/audio-text training workflows.
- **Model surgery**: depth expansion, width expansion, vocabulary expansion, knowledge grafting, and MoE upcycling.
- **Merge and export**: merge LoRA adapters, convert to GGUF, and generate Ollama Modelfiles.
- **Model editor**: manage identity, system prompts, knowledge base, safety guardrails, inference parameters, and model cards.
- **Runtime adaptation**: the launcher creates `.venv` and installs a PyTorch profile based on NVIDIA/CPU hardware.

## Quick Start

### 1. Requirements

Python **3.10+** is required. A short writable path is recommended, for example `C:\ForgeX` on Windows.

### 2. Launch

Windows:

```bat
run.bat
```

Linux / macOS:

```bash
chmod +x run.sh install.sh install_forgex.sh
./run.sh
```

On first launch, ForgeX creates `.venv` and installs dependencies automatically. The default UI address is:

```text
http://127.0.0.1:7860
```

### 3. Useful Commands

```bash
python launcher.py run             # install dependencies and launch UI
python launcher.py install         # install/update runtime only
python launcher.py fingerprint     # print machine fingerprint for legacy licensing
python launcher.py license-info    # show legacy license info; not required for OSS build
python selfcheck.py                # static self-check
python smoke_test.py               # lightweight smoke test
```

## Repository Layout

```text
ForgeX/
├─ core/                    # core training, distillation, export, dataset, and editing modules
├─ core/simulation/         # training/VRAM estimation utilities
├─ data/configs/            # default configuration
├─ data/datasets/           # user datasets; ignored by Git
├─ data/loras/              # training outputs; ignored by Git
├─ data/models_cache/       # model cache; ignored by Git
├─ docs/                    # architecture, roadmap, and open-source notes
├─ examples/                # sample datasets
├─ main.py                  # Gradio UI entry point
├─ launcher.py              # cross-platform launcher and runtime installer
├─ requirements.txt         # Python dependencies
└─ constraints.txt          # UI compatibility constraints
```

## Important Notes

- This repository does **not** include model weights or the large training dataset from the original user package.
- Training large models may require significant VRAM, disk space, and network bandwidth.
- When using third-party models, datasets, APIs, or papers, you are responsible for following their licenses and terms.
- Some model loading paths use `trust_remote_code=True`. Only load model repositories you trust.
- Do not commit API keys, private datasets, generated model weights, or commercial license files.

## Pre-release Checklist

- [x] Obfuscated core modules converted to readable source.
- [x] Removed `license.key`.
- [x] Removed large `train_*.jsonl` data.
- [x] Removed logs, temp files, and local model caches.
- [x] Added `.gitignore`, `LICENSE`, contribution guide, security notes, architecture docs, and sample data.
- [x] Static `python -m py_compile` check passed.

## License

This prepared open-source package uses the **MIT License** by default. Replace `LICENSE` before release if you prefer Apache-2.0, GPL-3.0, or a commercial license.
