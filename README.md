# WaveTag — 평가 프레임 & 견고성 테스트

WaveTag(비가청 음성 워터마킹 + 적대적 노이즈 결합) 보호 기술의 평가 프레임과 견고성 테스트 코드베이스.

- 기준 문서: 「WaveTag 기술 확정 정리 문서(도면/전체 플로우)」
- 참고 논문: Volto (Ge et al., IEEE TDSC 2026), EIAW (Liu et al., Big Data Mining and Analytics 2026)

> [!NOTE]
> - 모든 임계값·키·대역은 `configs/default.yaml` 한 곳에서 관리.
> - **★TBD★** 표시는 실험으로 확정할 값. 현재 값은 코드 실행용 임시값이므로 확정값으로 취급하지 말 것.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 전체 파이프라인 실행 (합성 데이터, Colab T4) | ✅ 확인 — 사전학습 모델(SpeechBrain, torchvision) 로딩 포함 |
| 음성 보호 성능 | ⚠️ 미검증 — 합성 음성에서 보호본 무음화 ([알려진 문제](#colab-실행으로-확인된-문제)) |
| 이미지 보호 성능 | ⚠️ 미검증 — 압축·전이 모델에 대한 효과 약함 ([알려진 문제](#colab-실행으로-확인된-문제)) |
| 실제 음성·이미지 데이터 평가 | ⏳ 미진행 |

검증 기준: 코드 커밋 `a1470e8`, [`notebooks/synthetic_e2e_check.ipynb`](notebooks/synthetic_e2e_check.ipynb)

## 목차

- [기술 스택](#기술-스택)
- [설치](#설치)
- [실행 방법](#실행-방법)
- [폴더 구조](#폴더-구조)
- [실행 환경별 지원 범위](#실행-환경별-지원-범위)
- [코드 ↔ 문서/논문 매핑](#코드--문서논문-매핑)
- [오픈소스 활용 현황](#오픈소스-활용-현황)
- [실험으로 확정할 값 (TBD)](#실험으로-확정할-값-tbd)
- [알려진 한계](#알려진-한계)
- [협업 규칙](#협업-규칙)

---

## 기술 스택

| 항목 | 선택 | 이유 |
|---|---|---|
| 언어 | **Python 3.11** | pesq, speechbrain 등의 휠·빌드 호환성이 가장 안정적 (3.10, Colab의 3.13에서도 동작) |
| 딥러닝 | **PyTorch 2.x** | 화자 인식 모델(SpeechBrain ECAPA-TDNN, x-vector), torchvision 분류기, 논문 평가 모델 6종 모두 PyTorch 계열 |
| 화자 모델 | **SpeechBrain** | `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/spkrec-xvect-voxceleb` 사전학습 모델 사용 |
| 신호 처리 | NumPy, SciPy | FFT, 필터, DCT (GPU 불필요) |
| 품질 지표 | pesq, pystoi, scikit-image | PESQ, STOI, PSNR, SSIM |
| 오류 정정 | reedsolo | Reed-Solomon (식별번호, 옵트아웃 코드) |
| 코덱 공격 | ffmpeg | MP3, AAC, Opus 인코딩 왕복 |

## 설치

### Google Colab (권장)

- [`notebooks/synthetic_e2e_check.ipynb`](notebooks/synthetic_e2e_check.ipynb)를 Colab에서 열고 `런타임 → 런타임 유형 변경 → T4 GPU` 선택 후 실행.
- PyTorch·ffmpeg 기본 설치 → `requirements.txt`만 추가 설치 (노트북에서 자동 수행).

### conda

ffmpeg 포함 설치.

```bash
conda env create -f environment.yml
conda activate wavetag
```

### venv + pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# GPU가 없는 환경은 CPU 휠을 먼저 설치 (생략하면 수 GB의 CUDA 라이브러리까지 설치됨)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

ffmpeg 별도 설치 필요.

| OS | 명령 |
|---|---|
| macOS | `brew install ffmpeg` |
| Windows | `winget install ffmpeg` |
| Ubuntu | `sudo apt install ffmpeg` |

### 참고 사항

- GPU 서버: CPU 휠 설치 생략, [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)에 따라 CUDA 버전에 맞게 설치.
- Windows `pesq` 설치: C 확장이라 Microsoft C++ Build Tools 필요. 어려우면 WSL2(Ubuntu) 권장.
  - `pesq` 미설치 시 PESQ는 `NaN` 처리 후 경고와 함께 계속 진행.
- Windows SpeechBrain 다운로드 중 심볼릭 링크 오류: 개발자 모드 활성화 또는 관리자 권한으로 실행.
- 설치 후 `pip freeze > requirements.lock.txt`로 팀 전체 패키지 버전을 고정할 것.

## 실행 방법

```bash
# 0) 단위 테스트 (인터넷·모델 불필요)
pytest -q

# 1) 스모크 테스트 — 코드 흐름만 확인 (수치는 의미 없음)
python scripts/make_synthetic_data.py
python scripts/run_audio_eval.py --data_dir data/synth_audio --out_dir results/smoke_audio --n_files 1 --dummy_encoder
python scripts/run_image_eval.py --data_dir data/synth_images --out_dir results/smoke_image --n_files 1 --no_pretrained

# 2) 실제 평가 (첫 실행 시 SpeechBrain, torchvision 가중치 자동 다운로드)
python scripts/run_audio_eval.py --data_dir data/vox50 --out_dir results/audio --n_files 10 --uid 1001 --strength strong
python scripts/run_image_eval.py --data_dir data/imagenet_val200 --out_dir results/image --n_files 50 --uid 1001

# 3) 모델 수준 검증 (학습 방해 + 소유권 검증)
python scripts/protect_dataset.py --data_dir data/spk --out_dir data/spk_protected --base_uid 1000
python scripts/train_sr_and_verify.py --clean_dir data/spk --protected_dir data/spk_protected --epochs 30

# 4) (선택) Volto* 사용 — configs에서 wm_mode: star, volto_star.ckpt 지정
python scripts/train_volto_star.py --data_dir data/vox50 --out ckpt/volto_star.pt --epochs 20
```

- 데이터 폴더 규약: [`data/README.md`](data/README.md) 참고.

> [!TIP]
> - 0)~4) 전체를 합성 데이터로 한 번에 실행: [`notebooks/synthetic_e2e_check.ipynb`](notebooks/synthetic_e2e_check.ipynb)
> - 노트북은 ACO 연산량을 줄인 `configs/colab.yaml`을 생성해 사용 (`default.yaml`은 변경하지 않음).

## 폴더 구조

```
configs/
  default.yaml              임계값·키·대역 설정 (TBD 포함)
wavetag/
  common/                   config, device, ecc(Reed-Solomon), payload(식별번호 직렬화), audio_io, certificate(인증서)
  audio/
    metrics.py              SNR, PESQ, STOI, 코사인 유사도
    speaker_encoder.py      SpeechBrain 화자 임베딩 (aux = 생성용, eval = 평가용)
    volto_removal.py        [도면 ③-A] 종방향 절삭(bisection) + 횡방향 절삭(ACO)
    volto_plus.py           [도면 ③-A] 시간영역 워터마크 + 데이터 수준 검출기
    volto_star.py           [도면 ③-A] 생성모델(U-Net G/D) 워터마크 (학습 필요)
    optout_code.py          [도면 ③-A] 고주파 DSSS 옵트아웃 코드 + RS ECC
    pipeline.py             ①~④ 전체 + ④-재처리 (보안강도 자동 조정)
  image/
    dct_utils.py            8x8 블록 DCT (NumPy, torch)
    modulo_embed.py         비밀키 p 모듈로 삽입·추출 (+ 키 디더)
    eiaw.py                 [도면 ③-B] 워터마크 + p 배수 적대적 최적화 + 옵트아웃
    models.py, metrics.py   torchvision 분류기, PSNR·SSIM
  robustness/
    audio_attacks.py        음성 공격 모음
    image_attacks.py        이미지 공격 모음
scripts/
  run_audio_eval.py         음성 평가 (보호 → 공격 → 지표 집계 → CSV)
  run_image_eval.py         이미지 평가 (화이트박스·전이 공격 성공률 포함)
  protect_dataset.py        데이터셋 일괄 보호 (모델 수준 검증 사전 단계)
  train_sr_and_verify.py    보호 데이터로 화자 인식 모델 학습 → 학습 방해·워터마크 반응 측정
  train_volto_star.py       Volto* 생성모델 학습
  make_synthetic_data.py    스모크 테스트용 합성 데이터 생성
notebooks/
  synthetic_e2e_check.ipynb 합성 데이터 전체 실행 검증 (Colab, 실행 결과 포함)
tests/                      단위 테스트 16개
```

## 실행 환경별 지원 범위

- `get_device()`: CUDA가 없으면 자동으로 CPU 사용. CPU 전용 최적화는 없음.

| 작업 | CPU | 비고 |
|---|---|---|
| 단위 테스트, 옵트아웃·Volto+·ECC·공격·인증서 | ✅ 즉시 | 수 초 (실측) |
| 이미지 보호·평가 (ResNet18, ≤256px, 수십 장) | ✅ 가능 | 장당 수 초~수십 초 (추정) |
| 음성 보호 파이프라인 (ACO, 5초 클립) | ⚠️ 느림 | ACO가 화자 인코더를 수백~수천 번 호출. 클립당 수 분~십수 분 (추정). `aco_max_candidates`, `aco_iters`, `freqs_per_step`으로 조절 |
| `train_sr_and_verify.py` (TinySR, 소규모) | ✅ 가능 | 점검용 소형 모델 |
| Volto\* 생성모델 학습 | ❌ 사실상 불가 | 전체 FFT(48,000점) 위 1D U-Net. CPU는 `--epochs 1` 동작만 확인 |
| 논문 수준 재현 (SR 모델 6종 × 데이터셋 3종, 수천 클립) | ❌ | GPU 서버 필요 |

- Colab T4: 논문 수준 재현을 제외한 모든 작업 실행 확인 (Volto\* 학습은 2 epoch 동작 확인).

## 코드 ↔ 문서/논문 매핑

| 도면 / 논문 | 코드 |
|---|---|
| ③-A 주파수 분해 + ACO 제거 (논문 Alg.1, 식 3~6) | `audio/volto_removal.py` |
| ③-A 식별번호 워터마크 (논문 Alg.2) | `audio/volto_plus.py`, `audio/volto_star.py` (식 7~9) |
| ③-A 고주파 옵트아웃 + ECC | `audio/optout_code.py`, `common/ecc.py` |
| ③-B DCT → 모듈로 → p 배수 최적화 → 옵트아웃 | `image/eiaw.py`, `image/modulo_embed.py` |
| ④ 검증 / ④-재처리 | `audio/pipeline.py` (`protect`) |
| ⑤ 인증서 | `common/certificate.py` |
| ⑦ 사후 검증 (음성: 모델 반응 / 이미지: 워터마크 추출) | `scripts/train_sr_and_verify.py` / `image.extract_image` |
| 4-2 쟁점 (1) 전이 성능 | `run_image_eval.py`의 `asr_train` vs `asr_held` |
| 4-2 쟁점 (3)(5) JPEG 후 유지·ECC | `image_attacks.py` + `wm_ok`, `oo_ok`, `*_ber` 열 |

## 오픈소스 활용 현황

| 구성 요소 | 공개 코드 | 처리 |
|---|---|---|
| **Volto** (논문 본체) | 저자 공식 코드 **찾지 못함** (2026-10-27 기준) | 논문 Sec.IV, Algorithm 1·2로 **직접 구현** (`audio/volto_*.py`). 논문에 없는 세부는 각 파일 상단에 "구현 가정"으로 명시 |
| **EIAW** | 공식 코드 **찾지 못함**. 유사 주제 저장소(예: "Generate-Adversarial-Examples-By-Digital-Watermarking")는 있으나 원 구현과 동일한지 **확인 못함** | 문서 설명(DCT + 모듈로 + p 배수 최적화)으로 **직접 구현** |
| 화자 임베딩 | SpeechBrain 사전학습 (ECAPA-TDNN, x-vector) | 그대로 사용 (`speaker_encoder.py`) |
| 이미지 분류기 | torchvision 사전학습 | 그대로 사용 |
| PESQ / STOI / PSNR / SSIM / RS | pesq, pystoi, scikit-image, reedsolo | 그대로 사용 |
| 비교 기준선 (권장) | Meta **AudioSeal** (음성 워터마킹) | 미포함. "기존 워터마킹 대비" 비교가 필요하면 추가 |

- 논문에 없는 구현 가정(A~D 등)은 부품별 검증·파라미터 탐색 등 자체 실험으로 확정.

## 실험으로 확정할 값 (TBD)

`configs/default.yaml`에서 관리.

| 구분 | 키 |
|---|---|
| 검증 합격 기준 | `pass_criteria` (`sim_max`, `pesq_min`, `snr_min_db`) |
| 보안강도 | `strength.*` (`target_sim`, `wm_amp_ratio`) |
| 음성 | 옵트아웃 대역·세기, `trigger_detect.min_peak_ratio`, `removal.pesq_floor` (현재 `null`, [알려진 문제](#colab-실행으로-확인된-문제) 참고) |
| 이미지 | `p`, 삽입 대역, `adv_*` |
| Volto\* | `volto_star.delta_scale` |

> [!IMPORTANT]
> PESQ는 `wb`/`nb` 모드에 따라 점수 범위가 다르다. 논문 기준값 1.99의 모드가 원문에 명시되어 있지 않으므로, 임계값은 반드시 같은 모드로 측정해 확정할 것.

## 알려진 한계

### Colab 실행으로 확인된 문제

[`notebooks/synthetic_e2e_check.ipynb`](notebooks/synthetic_e2e_check.ipynb) 실행(합성 데이터) 결과. 공격별 상세 수치는 노트북 참고.

1. **음성 보호 실패 시 무음화**
   - 실제 화자 모델로 보호 시 합성 음성 전부 화자 유사도 0.83~0.94 → 식별 임계값(`id_threshold` = 0.6) 미달.
   - 이때 `longitudinal_crop()`이 거의 모든 주파수 성분을 제거해 보호본이 무음에 가까워짐 (SNR ≈ 0.46 dB, PESQ ≈ 1.04).
   - 음질 하한 `removal.pesq_floor`가 `null`(비활성)이라 방지되지 않음.
   - 할 일: 실제 음성에서 재현 여부 확인, 보호 실패 시 원본을 지우지 않도록 수정.
2. **이미지 보호 효과 미확인**
   - 보호에 쓰지 않은 전이 모델(MobileNetV3) 기준 오분류 3장 중 1장.
   - 최적화 모델(ResNet18) 3장 전부 오분류는 보호 생성에 쓴 모델로 평가한 결과(화이트박스)라 보호 효과의 근거가 아님.
   - 워터마크는 JPEG 75, 옵트아웃 코드는 JPEG 90에서 복원 실패.
   - 평가 대상이 이미지 분류기라, 서비스가 막으려는 얼굴인식·생성형 AI 학습 방해와 다름 (문서 4-2 쟁점 2).
   - 할 일: 실제 이미지로 재측정, 다중 모델 최적화, 보호 대상에 맞는 평가 추가.
3. **인증서 발급 조건**
   - `run_audio_eval.py`가 검증 통과 여부와 관계없이 인증서 발급 (문서 ⑤는 기준 충족 시에만 발급).
   - 레포에는 인증서 검증 함수가 없음. 노트북에서 인감을 재계산하는 방식으로 위변조(인증서 내용·파일 변경) 검출 확인.
4. **Volto\* 워터마크 검출 미사용**
   - `volto_star.star_detect()`가 스크립트·테스트 어디에서도 호출되지 않음 → Volto\* 평가에 워터마크 검출 지표 없음.

### 구현 중 발견된 사실

1. **이미지 키 보안 결함 (수정 완료)**
   - 나머지(mod p)만으로 비트를 기록하면, 작은 계수에서는 값이 사실상 부호로 비트를 나타내 **틀린 p로도 복호**됨 (p=10~64 중 27개 전부).
   - 키 시드 기반 **디더** 추가로 해결 (`modulo_embed.py`). `test_embed_extract_lossless`가 회귀 방지.
2. **p가 작으면 JPEG 압축에 쉽게 깨짐**
   - 합성 이미지 1장 기준: p=8은 JPEG 95부터 워터마크·옵트아웃 모두 실패 (PSNR 43 dB).
   - p=16·24·32: 워터마크는 JPEG 75까지 복호, 옵트아웃(중·고주파)은 p=16·24에서 JPEG 95만, p=32에서 JPEG 90까지 생존 (PSNR 37 → 34 → 31 dB).
   - Colab 실행(합성 이미지 3장, p=16): 워터마크가 JPEG 90까지만 복호되어 위 결과와 다름.
   - **경계선 결과라 이미지·시드에 따라 달라짐 → 반드시 실제 이미지로 재측정할 것.**
   - 화질-생존성 트레이드오프(p↑ → PSNR↓), 옵트아웃이 워터마크보다 약한 점은 일관됨 → 문서 4-2 쟁점 (3)·(5)는 실제 위험.
3. **음성 옵트아웃 코드는 8 kHz 다운샘플·4 kHz 저역통과에서 소멸**
   - `test_optout_dies_under_8k_resample`과 Colab 실행 모두에서 확인.
   - MP3(64~192k), AAC, Opus, 잡음, 시간 이동, 양자화(10·8bit)에서는 생존 (10 dB 강한 잡음에서는 일부만 복원).
   - 전화망·저대역 코덱을 위협 모델에 포함할지 결정 필요.
4. **Volto+ 트리거는 LPC 백색화 후에만 검출됨**
   - 음성은 유색 신호라 단순 상관으로는 검출 불가.
   - 시간 이동 공격에서는 위치가 어긋나 `trigger_detected=0`.

### 논문·문서에 없어 가정한 부분

실험으로 검증 필요.

- ACO의 주파수 단위(좁은 대역 묶음, `band_hz`), 후보 풀 상한, 초기 가중치 정규화, 확률 정규화 — `volto_removal.py` 상단 가정 A~D.
- Volto\*의 G/D 동시 학습, 스펙트럼 2채널 U-Net 구조, 3초 단위 조각 처리.
- 이미지 DCT를 **8x8 블록 DCT**(JPEG 방식)로 구현 — EIAW 원 논문이 전체 DCT인지 블록 DCT인지 **확인하지 못함**.
- 적대적 최적화 규칙(그래디언트 상위 k% 계수를 ±p 이동, 조기 종료)은 자체 구현.

### 테스트 범위

- Colab(T4)에서 단위 테스트 16개 통과, `scripts/` 6개 전부를 **사전학습 모델(SpeechBrain, torchvision) + 합성 데이터**로 끝까지 실행 확인.
- **실제 음성·이미지 데이터로는 미실행.** 원본 `default.yaml` 설정(노트북은 ACO 축소 설정 사용)과 `very_strong` 시작 강도도 미실행.
- 화자 유사도·ASR은 **데이터 수준 대리 지표**. 학습 방해의 최종 근거는 `train_sr_and_verify.py`(또는 논문처럼 대형 화자 인식 모델 학습)의 정확도 하락으로 확인.
- 이미지는 학습 여부 검증(문서 4-2 쟁점 4) 불가 → 워터마크 추출(`extract_image`)로만 검증.

## 협업 규칙

- 함수·클래스에는 한국어 docstring, `★TBD★` · `가정` · `한계`는 반드시 주석에 표시.
- 새 임계값은 코드에 하드코딩하지 말고 `configs/default.yaml`에 추가.
- PR 전 `pytest -q` 통과.
- 평가 결과(`results/`)는 git에 올리지 않고 CSV 요약만 공유.
- `notebooks/`의 실행 결과 포함 노트북은 검증 기록용. 다시 실행해 갱신하면 [현재 상태](#현재-상태)의 기준 커밋도 함께 수정.
