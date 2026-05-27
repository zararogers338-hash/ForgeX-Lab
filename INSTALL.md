# ForgeX 安装说明 / Installation Guide

## 中文

### 系统要求

- Python 3.10 或更高版本
- Windows 10/11、Linux 或 macOS
- 训练大模型建议使用 NVIDIA GPU；CPU 也可以启动 UI 和处理轻量任务
- 足够的磁盘空间：模型、数据集和 LoRA 输出可能很大

### Windows

双击：

```bat
run.bat
```

或在命令行中运行：

```bat
python launcher.py run
```

只安装环境不启动 UI：

```bat
python launcher.py install
```

### Linux / macOS

```bash
chmod +x run.sh install.sh install_forgex.sh
./run.sh
```

只安装环境：

```bash
./install.sh
```

### 运行逻辑

启动器会：

1. 检查 Python 版本；
2. 创建 `.venv`；
3. 根据硬件选择 PyTorch 安装策略；
4. 安装 `requirements.txt`，并套用 `constraints.txt`；
5. 启动 Gradio UI。

### 常见问题

**Q: Windows 上虚拟环境创建失败怎么办？**  
A: 删除 `.venv` 后重试，或把项目移动到 `C:\ForgeX` 这类短路径、可写目录。

**Q: 需要 license.key 吗？**  
A: 开源版不需要。`license-info` 仅用于兼容旧用户版授权文件。

**Q: 为什么安装很慢？**  
A: PyTorch、Transformers、CUDA 依赖和模型文件都可能很大。建议使用稳定网络，并预留充足磁盘空间。

---

## English

### Requirements

- Python 3.10 or newer
- Windows 10/11, Linux, or macOS
- NVIDIA GPU is recommended for large-model training; CPU can still run the UI and lightweight tasks
- Sufficient disk space for models, datasets, and LoRA outputs

### Windows

Double-click:

```bat
run.bat
```

Or run:

```bat
python launcher.py run
```

Install runtime only:

```bat
python launcher.py install
```

### Linux / macOS

```bash
chmod +x run.sh install.sh install_forgex.sh
./run.sh
```

Install runtime only:

```bash
./install.sh
```

### What the Launcher Does

The launcher will:

1. Check the Python version;
2. Create `.venv`;
3. Select a PyTorch installation profile based on hardware;
4. Install `requirements.txt` with `constraints.txt`;
5. Start the Gradio UI.

### FAQ

**Q: Virtual environment creation fails on Windows.**  
A: Delete `.venv` and retry, or move the project to a short writable path such as `C:\ForgeX`.

**Q: Is license.key required?**  
A: No. The open-source build does not require it. `license-info` is kept only for legacy compatibility.

**Q: Why is installation slow?**  
A: PyTorch, Transformers, CUDA packages, and model files can be large. Use a stable connection and keep enough free disk space.
