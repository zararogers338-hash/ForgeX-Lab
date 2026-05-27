# 开源整理说明 / Open-source Preparation Notes

## 中文

本目录是从用户发布包整理出来的 GitHub 开源准备版，做过以下处理：

- 将轻量混淆的 Python 模块静态还原为源码；
- 删除 `license.key` 和旧的强制授权启动链路；
- 保留 `license_public.py` 作为旧授权文件兼容层，但开源版启动不依赖它；
- 删除大型训练数据、日志、临时文件和缓存目录内容；
- 保留必要目录并加入 `.gitkeep`；
- 增加中英双语文档与开源社区文件；
- 默认使用 MIT License。

### 发布前你还需要决定

- 是否真的使用 MIT License；
- 是否要把作者名从 `Chris Nuomi` 改成组织名；
- 是否要把旧授权相关文件完全移除；
- 是否要公开所有模型手术/蒸馏/导出功能，或拆分为社区版与商业版。

---

## English

This directory is a GitHub-ready open-source preparation build derived from the user release package. The following changes were made:

- Converted lightweight-obfuscated Python modules back into source code;
- Removed `license.key` and the mandatory legacy license startup path;
- Kept `license_public.py` as a legacy compatibility layer, but the open-source build does not depend on it;
- Removed large training data, logs, temporary files, and cache contents;
- Kept required directories with `.gitkeep`;
- Added bilingual documentation and open-source community files;
- Defaulted to the MIT License.

### Decisions before publishing

- Whether MIT License is the license you want;
- Whether to replace `Chris Nuomi` with an organization name;
- Whether to fully remove legacy licensing files;
- Whether all model surgery/distillation/export features should be public, or split into community/commercial editions.
