# 贡献指南 / Contributing Guide

## 中文

欢迎贡献代码、文档、示例数据、Bug 报告和功能建议。

### 开发流程

1. Fork 仓库；
2. 创建分支：`git checkout -b feature/your-feature`；
3. 修改代码并运行检查：

```bash
python selfcheck.py
python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")
```

4. 提交 Pull Request，并说明变更内容、测试方式和潜在风险。

### 代码原则

- 不提交模型权重、用户数据集、API Key、license.key 或私钥。
- 新增功能应尽量保持本地运行、可配置、可回滚。
- 涉及模型下载、远程代码加载、文件删除、外部命令执行时，请在文档中说明风险。
- UI 文案建议中英双语，或至少保证关键操作含义清楚。

---

## English

Contributions are welcome: code, documentation, sample data, bug reports, and feature requests.

### Workflow

1. Fork the repository;
2. Create a branch: `git checkout -b feature/your-feature`;
3. Run checks:

```bash
python selfcheck.py
python -m py_compile $(find . -name "*.py" -not -path "./.venv/*")
```

4. Open a Pull Request and describe the change, test method, and potential risks.

### Principles

- Do not commit model weights, private datasets, API keys, `license.key`, or private keys.
- New features should remain local-first, configurable, and reversible where possible.
- Document risks when changes involve model downloads, remote code loading, file deletion, or external command execution.
- UI text should ideally be bilingual, or at least clear for critical actions.
