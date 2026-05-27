# 更新日志 / Changelog

## v3.2.0-open-source — 2026-05-28

### 中文

- 将用户发布包整理为适合 GitHub 开源的仓库结构。
- 将 35 个轻量混淆的核心模块还原为可读源码。
- 开源版默认取消 `license.key` 强制校验，可直接启动。
- 修复 `launcher.py install` 命令缺失的问题。
- 删除试运行 `license.key`、大型训练数据、日志和临时文件。
- 新增中英双语 README、安装说明、贡献指南、安全说明、行为准则、架构文档、路线图和示例数据。
- 扩展 `.gitignore`，避免提交模型权重、LoRA 输出、用户数据、缓存和私钥。
- 静态编译检查通过。

### English

- Reorganized the user release package into a GitHub-ready open-source repository.
- Converted 35 lightweight-obfuscated core modules back into readable source code.
- Removed mandatory `license.key` checks for the open-source build.
- Fixed the missing `launcher.py install` command.
- Removed trial `license.key`, large training data, logs, and temporary files.
- Added bilingual README, installation guide, contribution guide, security policy, code of conduct, architecture docs, roadmap, and sample data.
- Expanded `.gitignore` to avoid committing model weights, LoRA outputs, user data, caches, and private keys.
- Static Python compilation check passed.
