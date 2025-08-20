# 1) 定位 fairseq 配置文件（路径名里那串hash每人可能不同；用通配符）
FSEQ_DIR=/mnt/workspace/AISumerCamp_audio_generation_fight/SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1/fairseq/dataclass
CFG=$FSEQ_DIR/configs.py

# 2) 先把 dataclasses 的导入加上 field（若已有则无影响）
sed -i.bak 's/from dataclasses import dataclass/from dataclasses import dataclass, field/' "$CFG"

# 3) 把所有  XxxxConfig() 这种“可变默认值”替换成  field(default_factory=XxxxConfig)
sed -E -i.bak 's/=\s*([A-Za-z_]+Config)\(\)/= field(default_factory=\1)/g' "$CFG"

# 4) 快速自检：仅测试导入，不跑模型
PYTHONPATH=/mnt/workspace/AISumerCamp_audio_generation_fight/SSL_Anti-spoofing \
/mnt/workspace/AISumerCamp_audio_generation_fight/.venvs/assis/bin/python - <<'PY'
import importlib
m = importlib.import_module("model")  # 根目录的 model.py
print("Loaded:", m)
print("Has Model:", hasattr(m, "Model"))
PY

# 5) 跑一次 CLI（仍然使用你的 ckpt 和 wav）
/mnt/workspace/AISumerCamp_audio_generation_fight/.venvs/assis/bin/python \
  utils/aasist_score_cli.py \
  --repo /mnt/workspace/AISumerCamp_audio_generation_fight/SSL_Anti-spoofing \
  --ckpt /mnt/workspace/AISumerCamp_audio_generation_fight/models/AASIST/model/Best_LA_model_for_DF.pth \
  --wav  /mnt/workspace/AISumerCamp_audio_generation_fight/MyLife1906-result/1.wav
