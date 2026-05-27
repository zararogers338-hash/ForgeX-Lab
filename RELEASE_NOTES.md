# ForgeX v3.2.0-open-source Release Notes

## 中文

本版本是面向 GitHub 开源发布的整理版。它不是原来的授权用户包，而是便于公开审计、学习和二次开发的源码包。

### 关键变化

- 开源版默认不需要 `license.key`。
- 允许直接运行 `python main.py`，也可以使用 `run.bat` / `run.sh`。
- 核心模块已从轻量混淆包装还原为可读 Python 源码。
- 删除试运行授权文件、大型训练数据、日志、缓存和临时文件。
- 增加中英双语项目文档和 GitHub 开源辅助文件。

### 发布建议

发布到 GitHub 前，请确认你接受当前 `LICENSE` 中的 MIT 授权；如果不接受，请先更换许可证。

---

## English

This is a GitHub-ready open-source preparation release. It is not the original licensed user package; it is a readable source package for public review, learning, and further development.

### Key Changes

- `license.key` is not required by default.
- Direct `python main.py` launch is allowed, and `run.bat` / `run.sh` still work.
- Core modules were converted from lightweight obfuscation wrappers back into readable Python source.
- Removed trial license file, large training data, logs, caches, and temporary files.
- Added bilingual project documentation and GitHub open-source helper files.

### Release Advice

Before publishing, confirm that you accept the current MIT license in `LICENSE`. Replace it first if you prefer a different license.
