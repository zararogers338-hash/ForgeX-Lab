# GitHub 发布步骤 / GitHub Publishing Steps

## 中文

在发布前，先确认 `LICENSE` 是否就是你想采用的开源许可证。

```bash
git init
git add .
git commit -m "Initial open-source release of ForgeX"
git branch -M main
git remote add origin https://github.com/<your-name-or-org>/ForgeX.git
git push -u origin main
```

如果仓库已经存在：

```bash
git remote add origin https://github.com/<your-name-or-org>/ForgeX.git
git push -u origin main
```

建议发布后立刻检查 GitHub 页面，确认没有上传以下内容：

- `license.key`
- `.env`
- API Key 或私钥
- 用户数据集
- 模型权重、LoRA 输出、GGUF 文件
- 本地缓存、日志和临时文件

---

## English

Before publishing, confirm that `LICENSE` is the open-source license you actually want.

```bash
git init
git add .
git commit -m "Initial open-source release of ForgeX"
git branch -M main
git remote add origin https://github.com/<your-name-or-org>/ForgeX.git
git push -u origin main
```

If the remote repository already exists:

```bash
git remote add origin https://github.com/<your-name-or-org>/ForgeX.git
git push -u origin main
```

After publishing, check the GitHub page and make sure the following were not uploaded:

- `license.key`
- `.env`
- API keys or private keys
- user datasets
- model weights, LoRA outputs, GGUF files
- local caches, logs, and temporary files
