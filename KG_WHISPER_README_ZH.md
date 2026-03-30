# KG-Whisper: 关键词引导的 Whisper + AdaKWS

复现 aiOla 的两篇论文:
- **AdaKWS**: [Adaptive Keyword Spotting (arXiv:2309.08561)](https://arxiv.org/abs/2309.08561)
- **KG-Whisper-PT**: [Keyword-Guided Adaptation of ASR (arXiv:2406.02649)](https://arxiv.org/abs/2406.02649)

## 概述

关键词引导语音识别的两阶段流水线:

1. **AdaKWS** - 使用 Whisper Small 编码器 + Character LSTM + AdaIN 的开放词汇关键词检测
2. **KG-Whisper-PT** - 使用 12 个学习的 prefix 向量（15K 参数）引导 Whisper Large-v2 解码器的提示调优

```
音频 + 关键词列表 → AdaKWS（检测关键词）→ KG-Whisper-PT（引导转录）
```

## 复现结果

### AdaKWS（VoxPopuli EN 测试集，1842 样本）

| 指标 | 复现结果 | 论文参考值 |
|------|---------|-----------|
| F1（阈值 0.3） | **96.17%** | 96.3% |
| AUC | **98.64%** | - |
| EER | **3.56%** | - |

### KG-Whisper-PT（VoxPopuli EN 测试集，1842 样本）

| 方法 | WER |
|------|-----|
| 原始 Whisper Large-v2 | 7.50% |
| Baseline（PT，无关键词） | 7.15% |
| Combined（AdaKWS + PT） | **7.06%** |
| Oracle（完美关键词） | 6.91% |

### VoxPopuli 详细统计（1842 样本）

| 分类 | 数量 | 说明 |
|------|------|------|
| 改善 | 110 | Combined WER < Baseline WER |
| 不变 | 1658 | Combined WER = Baseline WER |
| - 两者 WER = 0% | 707 | 所有模型正确，无干扰 |
| - 两者 WER > 0% | 951 | 有无关键词错误相同 |
| 变差 | 74 | Combined WER > Baseline WER（负采样噪声导致） |

- Baseline WER = 0%: 731 / 1842 (39.7%)
- Combined WER = 0%: 754 / 1842 (40.9%) — 关键词引导额外修正了 23 个样本到完美

### 跨域: Medical ASR（自定义关键词列表，80 个医学术语）

| 方法 | WER（Top 20 样本） |
|------|-------------------|
| Baseline（无关键词） | 50.45% |
| Combined（使用关键词列表） | **27.35%** |

- 20 个中 11 个改善，8 个修正到 WER 0%
- 零样本跨域: 在 VoxPopuli 上训练，无需微调直接在医学领域测试

### 训练配置

- **AdaKWS v3**: 英语，25 个 epoch，跨音频负采样，Whisper Small 编码器
- **KG-Whisper-PT v2**: 英语，30K 步，batch 内负采样，Whisper Large-v2 解码器，12 个 prefix 向量

## 安装

### 环境要求

- Python 3.9+
- CUDA GPU（推荐: 8GB+ 显存）

### 使用 uv 安装

```bash
# 安装 uv（如未安装）
pip install uv

# 克隆仓库
git clone https://github.com/YOUR_USERNAME/kg-whisper.git
cd kg-whisper

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装项目
uv pip install -e .

# 安装额外依赖
uv pip install gradio soundfile datasets nltk
```

### 下载权重

从 Google Drive 下载: [权重文件](https://drive.google.com/drive/folders/1MaDFDu-aUwNVy3EuxEDdVL6ShfrXog9D?usp=sharing), [Demo 音频](https://drive.google.com/drive/folders/1zqFEOnfrXyP2h9bCijdYqCVvT3VaCZgk?usp=sharing)

放置如下:
```
kg_whisper/outputs/
├── checkpoint_final.pt                          # KG-Whisper-PT (183K)
└── adakws_v3/
    └── adakws_checkpoint_29000.pt               # AdaKWS v3 (261MB)
```

### 下载 Demo 音频

从 Google Drive 下载: [权重文件](https://drive.google.com/drive/folders/1MaDFDu-aUwNVy3EuxEDdVL6ShfrXog9D?usp=sharing), [Demo 音频](https://drive.google.com/drive/folders/1zqFEOnfrXyP2h9bCijdYqCVvT3VaCZgk?usp=sharing)

放置如下:
```
demo/
├── audio_best/                    # VoxPopuli 最佳 20 样本
│   ├── best_00.wav ... best_19.wav
│   └── ground_truth.json
├── audio_medical_best/            # Medical 最佳 20 样本
│   ├── medical_best_00.wav ... medical_best_19.wav
│   ├── ground_truth.json
│   └── medical_keywords.txt
└── medical_keywords.txt           # 80 个医学术语
```

## 运行 Demo

```bash
python -m demo.app \
    --device cuda \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --whisper_pt_checkpoint kg_whisper/outputs/checkpoint_final.pt
```

在浏览器中打开 http://localhost:7860

### Demo 功能

| 标签页 | 说明 |
|--------|------|
| **Single Sample** | 选择样本，运行完整流水线（Baseline / AdaKWS / Combined / Oracle） |
| **Custom Keywords** | 上传音频 + 自定义关键词列表（真实场景模式） |
| **Batch Evaluation** | 运行所有样本，可选自定义关键词列表，下载 CSV |

### Demo 可用数据集

| 数据集 | 说明 |
|--------|------|
| Best Samples | VoxPopuli Top 20（改善最大） |
| Worst Samples | VoxPopuli 80 个样本（Combined > Baseline） |
| Medical Best | Medical Top 20（跨域，改善最大） |
| Medical ASR | 10 个医学样本 |

## 训练

### 训练 KG-Whisper-PT

```bash
python -m kg_whisper.train \
    --device cuda \
    --cache_dir "YOUR_CACHE_DIR" \
    --total_steps 30000 \
    --batch_size 4
```

### 训练 AdaKWS

```bash
python -m kg_whisper.adakws_train \
    --device cuda \
    --cache_dir "YOUR_CACHE_DIR" \
    --epochs 25
```

## 评估

### 联合评估（VoxPopuli）

```bash
python -m kg_whisper.eval_combined \
    --pt_checkpoint kg_whisper/outputs/checkpoint_final.pt \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --device cuda --cache_dir "YOUR_CACHE_DIR" --mode all
```

### AdaKWS F1/AUC/EER

```bash
python -m kg_whisper.adakws_eval \
    --checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --device cuda --cache_dir "YOUR_CACHE_DIR"
```

## 项目结构

```
kg_whisper/
├── model.py              # KGWhisperPT 模型（prefix tuning）
├── config.py             # 配置
├── data.py               # 数据集和数据加载器
├── train.py              # KG-Whisper-PT 训练
├── adakws_model.py       # AdaKWS 模型（Whisper 编码器 + CharLSTM + AdaIN）
├── adakws_data.py        # AdaKWS 数据集和负采样
├── adakws_train.py       # AdaKWS 训练（含跨音频负采样）
├── adakws_eval.py        # AdaKWS 评估（F1/AUC/EER）
├── eval_combined.py      # 完整流水线评估
├── eval_medical.py       # 医学领域评估
├── kws_simulator.py      # 关键词采样模拟器
└── outputs/              # 权重文件（需单独下载）

demo/
├── app.py                # Gradio Demo 应用
├── find_best_samples.py  # 查找 VoxPopuli 最佳样本
├── find_worst_samples.py # 查找 VoxPopuli 最差样本
├── find_best_medical_samples.py   # 查找 Medical 最佳样本
├── find_worst_medical_samples.py  # 查找 Medical 最差样本
├── precompute_keywords.py         # 预计算关键词以稳定结果
├── stats_voxpopuli.py             # VoxPopuli 统计
└── medical_keywords.txt           # 医学关键词列表（80 个术语）
```

## 核心结论

1. **AdaKWS F1 96.17%** 接近论文 96.3% — 关键词检测成功复现
2. **Combined < Baseline**（7.06% < 7.15%）— 关键词引导改善了转录
3. **跨域有效** — 在 VoxPopuli 上训练，无需微调即可改善 Medical ASR
4. **自定义关键词列表** — 推荐 10-50 个领域专业术语（aiOla 官方建议）
5. **音频质量是关键** — 清晰音频 + 关键词列表 = 最佳效果；音频质量差则任何模型都无能为力

## 参考文献

- [AdaKWS: Adaptive Keyword Spotting](https://arxiv.org/abs/2309.08561)
- [KG-Whisper: Keyword-Guided Adaptation of ASR](https://arxiv.org/abs/2406.02649)
- [aiOla Jargonic](https://aiola.ai/jargonic/)
- [OpenAI Whisper](https://github.com/openai/whisper)
