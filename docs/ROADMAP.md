# 路线图 / Roadmap

## 中文

优先级建议：

1. **稳定性**：把训练、导出、模型编辑等核心流程拆出更小的服务层，减少 `main.py` 体积。
2. **测试**：为数据转换、质量分析、参数推荐、模型路径解析增加单元测试。
3. **安全**：给 `trust_remote_code`、外部命令、文件删除、下载脚本增加更明确的 UI 提示。
4. **文档**：补充每个训练模式的最小可运行教程。
5. **插件化**：把导出器、训练器、数据转换器做成可插拔接口。
6. **国际化**：把 UI 文案抽离为 i18n 字典。
7. **任务恢复**：增加训练中断后的状态恢复与日志追踪。

---

## English

Suggested priorities:

1. **Stability**: split training, export, and model-editing flows into smaller service layers to reduce `main.py` size.
2. **Testing**: add unit tests for dataset conversion, quality analysis, parameter recommendations, and model path resolution.
3. **Security**: add clearer UI warnings for `trust_remote_code`, external commands, file deletion, and downloaded scripts.
4. **Documentation**: add minimal runnable tutorials for each training mode.
5. **Plugin architecture**: make exporters, trainers, and dataset converters pluggable.
6. **Internationalization**: extract UI text into i18n dictionaries.
7. **Task recovery**: add resumable training state and better log tracing.
