# AutoDL 部署手册

目标平台：AutoDL (RTX 5090 / 5090 D, CUDA 12.8, Python 3.12)

---

## 架构

```
mineru-openai-server  :30000   (vLLM gateway, VLM 模型推理)
        ↓
mineru-api            :8000    (MinerU REST API)
        ↓
mineru-mcp            :6006    (FastMCP SSE, AutoDL 自定义服务端口)
```

---

## 1. 环境准备

```bash
# 不要开学术加速 —— 会破坏 Aliyun 内网镜像 172.20.0.113
# 只在需要 GitHub 拉取时临时开启，pip 安装时关闭
```

---

## 2. 安装 MinerU（必须从源码安装 2.7.0，不能直接 pip install）

> **关键**：直接 `pip install mineru[all]` 会拉取 vLLM 0.21.0，
> 该版本对 SM 12.x (Blackwell) 要求 CUDA ≥ 12.9，而 AutoDL 是 CUDA 12.8，会报错：
> `SM 12.x requires CUDA >= 12.9` → `Engine core initialization failed`
>
> 从源码安装 `mineru-2.7.0-released` 分支会固定安装 vLLM 0.10.1.1 (cu128)，兼容 CUDA 12.8。

```bash
# 开学术加速拉取 GitHub
source /etc/network_turbo

git clone https://github.com/opendatalab/MinerU.git
cd MinerU
git checkout mineru-2.7.0-released

# 关闭学术加速，改用清华镜像安装依赖（Aliyun 镜像有哈希校验问题）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
pip install -e .[all] -i https://pypi.tuna.tsinghua.edu.cn/simple

cd ~
```

---

## 3. 下载模型（用 ModelScope，速度快）

MinerU 首次启动会自动从 ModelScope 下载，也可提前下载：

```bash
# Pipeline 模型（布局检测等）
mineru-models-download -s modelscope -m pipeline

# VLM 模型（MinerU2.5-Pro）
mineru-models-download -s modelscope -m vlm
```

模型缓存路径：`/root/.cache/modelscope/hub/models/OpenDataLab/`

---

## 4. 安装 mineru-mcp

```bash
# 开学术加速拉取 GitHub
source /etc/network_turbo

git clone https://github.com/<your-org>/mineru-mcp.git ~/mineru-mcp

# 关闭学术加速
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

pip install fastmcp "uvicorn[standard]" aiohttp python-dotenv \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 5. 配置 .env

```bash
cp ~/mineru-mcp/.env.example ~/mineru-mcp/.env
```

编辑 `~/mineru-mcp/.env`，关键配置：

```env
# AutoDL 自定义服务端口（必须是 6006，才能从外部访问）
MCP_PORT=6006

MINERU_API_URL=http://localhost:8000
MINERU_BACKEND=vlm-http-client
MINERU_VLM_HTTP_URL=http://localhost:30000
MINERU_LANG=ch
MINERU_TIMEOUT_SEC=1800
MINERU_WORK_DIR=/root/autodl-tmp/mineru_mcp
```

```bash
mkdir -p /root/autodl-tmp/mineru_mcp
```

---

## 6. 用 tmux 启动三个服务

```bash
tmux new-session -s mineru
```

### 窗口 1 — VLM 推理服务

```bash
# tmux 窗口重命名：Ctrl+B, ,  → 输入 vlm
MINERU_MODEL_SOURCE=modelscope mineru-openai-server --port 30000
```

等待日志出现 `Application startup complete` 再启动下一个。

### 窗口 2 — MinerU API

```bash
# Ctrl+B, c  新建窗口
mineru-api --host 127.0.0.1 --port 8000
```

> 必须绑定 `127.0.0.1`，绑定 `0.0.0.0` 时 mineru-api 会因 SSRF 风险禁用 vlm-http-client 后端。

### 窗口 3 — MCP 服务

```bash
# Ctrl+B, c  新建窗口
cd ~/mineru-mcp
python server.py
```

---

## 7. 验证

```bash
# 检查三个进程
curl http://localhost:30000/health
curl http://localhost:8000/docs
curl http://localhost:6006/sse   # 应返回 SSE 事件流头
```

AutoDL 自定义服务外部访问地址（在 AutoDL 控制台「自定义服务」查看）：
`https://<实例域名>:8443/sse`

---

## 常见问题

| 错误 | 原因 | 修复 |
|------|------|------|
| `SM 12.x requires CUDA >= 12.9` | vLLM 版本太新 (0.21.0+) | 必须从 `mineru-2.7.0-released` 源码安装 |
| `LocalEntryNotFoundError` (VLM 模型) | 从 HuggingFace 找模型，但网络不通 | 加 `MINERU_MODEL_SOURCE=modelscope` 启动 |
| `vlm-http-client backend disabled` | mineru-api 绑定了 0.0.0.0 | 改为 `--host 127.0.0.1` |
| pip 哈希校验失败 | 学术加速与 Aliyun 镜像冲突 | pip 安装时关闭学术加速，用清华镜像 |
| MCP 外部无法访问 | 端口不是 6006 | `.env` 里 `MCP_PORT=6006` |
| `GET /v1/models` 返回 500 | `prometheus_fastapi_instrumentator` 与 FastAPI 版本不兼容 | 执行以下 patch：`sed -i 's/route_name = route\.path/route_name = getattr(route, "path", None)/' /root/miniconda3/lib/python3.12/site-packages/prometheus_fastapi_instrumentator/routing.py`，然后重启 vLLM |
| vLLM 重启报 `Free memory less than desired` | 上次进程被 Ctrl+C 后 GPU 显存未完全释放 | `pkill -f vllm && sleep 3`，再加 `--gpu-memory-utilization 0.45` 重启 |
| 文件名太长报错 | 中文文件名 URL 编码后超出 Linux 限制 | `cp` 成短文件名后再解析 |
