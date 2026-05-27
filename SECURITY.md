# 安全说明 / Security Policy

## 中文

ForgeX 是本地 AI 训练工具，可能处理模型权重、数据集、API Key 和用户文档。请注意：

- 不要把隐私数据、商业数据或受版权限制的数据直接提交到仓库。
- 不要提交 `.env`、`license.key`、API Key、私钥或个人模型缓存。
- 只加载你信任的模型仓库；部分流程可能使用 `trust_remote_code=True`。
- 训练和导出流程可能调用外部命令、下载转换脚本或写入大量文件。
- 使用第三方 API 进行蒸馏时，请遵守服务条款和数据政策。
- 生成模型可能输出错误、有害或侵权内容；发布前请进行安全评估。

如发现安全问题，请优先私下联系维护者，不要直接公开利用细节。

---

## English

ForgeX is a local AI training tool and may process model weights, datasets, API keys, and user documents. Please note:

- Do not commit private data, commercial data, or copyrighted datasets.
- Do not commit `.env`, `license.key`, API keys, private keys, or local model caches.
- Only load model repositories you trust; some paths may use `trust_remote_code=True`.
- Training/export workflows may invoke external commands, download conversion scripts, or write large files.
- When using third-party APIs for distillation, follow their terms and data policies.
- Generated models may produce inaccurate, harmful, or infringing outputs; evaluate before release.

If you find a security issue, please contact the maintainer privately first instead of publishing exploit details.
