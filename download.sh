#!/usr/bin/env bash
set -euo pipefail

# 建议放到一个临时目录
WORKDIR="$(pwd)/piper_cn_download"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 独立环境，避免污染你比赛环境
python3 -m venv .venv/download && source .venv/download/bin/activate
python -m pip install --upgrade pip
# 只装下载工具，不装推理库
python -m pip install "huggingface_hub>=0.23.0"

# 允许 Hugging Face 国内镜像（可选）
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# 用 huggingface_hub 精确拉取所需文件
python - <<'PY'
from huggingface_hub import snapshot_download
repo_id = "rhasspy/piper-voices"
# 只拉我们要的中文 huayan 两档（onnx+json）
allow = [
    "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
    "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
    "zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx",
    "zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json",
]
snapshot_download(repo_id=repo_id, allow_patterns=allow, local_dir="piper_zh_CN_bundle")
print("Downloaded to piper_zh_CN_bundle/")
PY

# 统一放到 bundle 根目录
mkdir -p bundle/models/piper
cp -f piper_zh_CN_bundle/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx* bundle/models/piper/
cp -f piper_zh_CN_bundle/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx* bundle/models/piper/

# 附带一个 readme
cat > bundle/models/piper/README.txt <<'TXT'
This bundle contains Piper Mandarin voices:
- zh_CN-huayan-medium.onnx (+ .json) 22.05kHz (better quality)
- zh_CN-huayan-x_low.onnx (+ .json) 16kHz (fastest)
Place them anywhere; keep the .onnx and .json side-by-side.
TXT

# 打包
# tar czf piper_zh_CN_huayan_bundle.tgz -C bundle .
# echo "Created: $(pwd)/piper_zh_CN_huayan_bundle.tgz"
