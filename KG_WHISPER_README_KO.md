# KG-Whisper: 키워드 가이드 Whisper + AdaKWS

aiOla의 두 논문 재현:
- **AdaKWS**: [Adaptive Keyword Spotting (arXiv:2309.08561)](https://arxiv.org/abs/2309.08561)
- **KG-Whisper-PT**: [Keyword-Guided Adaptation of ASR (arXiv:2406.02649)](https://arxiv.org/abs/2406.02649)

## 개요

키워드 가이드 음성 인식을 위한 2단계 파이프라인:

1. **AdaKWS** - Whisper Small 인코더 + Character LSTM + AdaIN을 사용한 개방형 키워드 탐지
2. **KG-Whisper-PT** - 12개의 학습된 prefix 벡터(15K 파라미터)로 Whisper Large-v2 디코더를 가이드하는 프롬프트 튜닝

```
오디오 + 키워드 목록 → AdaKWS (키워드 탐지) → KG-Whisper-PT (가이드 전사)
```

## 재현 결과

### AdaKWS (VoxPopuli EN 테스트, 1842 샘플)

| 지표 | 재현 결과 | 논문 참조값 |
|------|----------|-----------|
| F1 (임계값 0.3) | **96.17%** | 96.3% |
| AUC | **98.64%** | - |
| EER | **3.56%** | - |

### KG-Whisper-PT (VoxPopuli EN 테스트, 1842 샘플)

| 방법 | WER |
|------|-----|
| 순수 Whisper Large-v2 | 7.50% |
| Baseline (PT, 키워드 없음) | 7.15% |
| Combined (AdaKWS + PT) | **7.06%** |
| Oracle (완벽한 키워드) | 6.91% |

### VoxPopuli 상세 통계 (1842 샘플)

| 분류 | 수량 | 설명 |
|------|------|------|
| 개선됨 | 110 | Combined WER < Baseline WER |
| 동일 | 1658 | Combined WER = Baseline WER |
| - 둘 다 WER = 0% | 707 | 모든 모델 정확, 간섭 없음 |
| - 둘 다 WER > 0% | 951 | 키워드 유무와 관계없이 동일한 오류 |
| 악화됨 | 74 | Combined WER > Baseline WER (네거티브 샘플링 아티팩트) |

- Baseline WER = 0%: 731 / 1842 (39.7%)
- Combined WER = 0%: 754 / 1842 (40.9%) — 키워드로 23개 추가 샘플이 완벽하게 수정됨

### 크로스 도메인: Medical ASR (커스텀 키워드 목록, 80개 의료 용어)

| 방법 | WER (상위 20 샘플) |
|------|-------------------|
| Baseline (키워드 없음) | 50.45% |
| Combined (키워드 목록 사용) | **27.35%** |

- 20개 중 11개 개선, 8개는 WER 0%로 수정
- 제로샷 크로스 도메인: VoxPopuli로 학습, 파인튜닝 없이 Medical에서 테스트

### 학습 구성

- **AdaKWS v3**: 영어, 25 에폭, 크로스 오디오 네거티브 샘플링, Whisper Small 인코더
- **KG-Whisper-PT v2**: 영어, 30K 스텝, 배치 내 네거티브 샘플링, Whisper Large-v2 디코더, 12개 prefix 벡터

## 설치

### 요구 사항

- Python 3.9+
- CUDA GPU (권장: 8GB+ VRAM)

### uv로 설치

```bash
# uv 설치 (미설치 시)
pip install uv

# 저장소 클론
git clone https://github.com/YOUR_USERNAME/kg-whisper.git
cd kg-whisper

# 가상환경 생성 및 의존성 설치
uv venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 프로젝트 설치
uv pip install -e .

# 추가 의존성 설치
uv pip install gradio soundfile datasets nltk
```

### 체크포인트 다운로드

Google Drive에서 다운로드: [체크포인트](https://drive.google.com/drive/folders/1MaDFDu-aUwNVy3EuxEDdVL6ShfrXog9D?usp=sharing), [데모 오디오](https://drive.google.com/drive/folders/1zqFEOnfrXyP2h9bCijdYqCVvT3VaCZgk?usp=sharing)

다음과 같이 배치:
```
kg_whisper/outputs/
├── checkpoint_final.pt                          # KG-Whisper-PT (183K)
└── adakws_v3/
    └── adakws_checkpoint_29000.pt               # AdaKWS v3 (261MB)
```

## 데모 실행

```bash
python -m demo.app \
    --device cuda \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --whisper_pt_checkpoint kg_whisper/outputs/checkpoint_final.pt
```

브라우저에서 http://localhost:7860 을 엽니다.

### 데모 기능

| 탭 | 설명 |
|----|------|
| **Single Sample** | 샘플 선택, 전체 파이프라인 실행 (Baseline / AdaKWS / Combined / Oracle) |
| **Custom Keywords** | 오디오 업로드 + 커스텀 키워드 목록 (실제 사용 모드) |
| **Batch Evaluation** | 전체 샘플 실행, 커스텀 키워드 목록 옵션, CSV 다운로드 |

## 주요 발견

1. **AdaKWS F1 96.17%**는 논문의 96.3%와 일치 — 키워드 탐지 성공적으로 재현
2. **Combined < Baseline** (7.06% < 7.15%) — 키워드 가이드가 전사를 개선
3. **크로스 도메인 작동** — VoxPopuli로 학습, 파인튜닝 없이 Medical ASR 개선
4. **커스텀 키워드 목록** — 도메인별 10-50개 용어 권장 (aiOla 문서 기준)
5. **오디오 품질이 핵심** — 깨끗한 오디오 + 키워드 목록 = 최상의 결과

## 참조

- [AdaKWS: Adaptive Keyword Spotting](https://arxiv.org/abs/2309.08561)
- [KG-Whisper: Keyword-Guided Adaptation of ASR](https://arxiv.org/abs/2406.02649)
- [aiOla Jargonic](https://aiola.ai/jargonic/)
- [OpenAI Whisper](https://github.com/openai/whisper)
