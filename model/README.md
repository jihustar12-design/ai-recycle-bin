# 모델: 공개 데이터(TrashNet)로 학습

## 1. 데이터셋 만들기

```bash
cd ~/Desktop/피지컬ai/ai-recycle-bin
python3 -m pip install pillow        # 처음 한 번
python3 model/prepare_dataset.py
```

결과: `model/dataset/train/` 1,640장, `model/dataset/test/` 410장 (클래스당 410장, 8:2 분할).
파일명 맨 앞이 라벨이에요 (`can.metal12.jpg`). 이 폴더는 `.gitignore` 처리되어 GitHub에 올라가지 않아요.

| 라벨 | 원본 (TrashNet) | 칸 |
|---|---|---|
| can | metal | 0 |
| pet | plastic | 1 |
| paper | paper + cardboard | 2 |
| general | trash + glass | 3 |
| empty | 사진의 빈 배경을 잘라 합성 | (동작 안 함) |

출처: TrashNet — Gary Thung, Mindy Yang (Stanford CS229, MIT License) https://github.com/garythung/trashnet

## 2. Edge Impulse 학습

1. https://studio.edgeimpulse.com 가입 → **Create new project** (`ai-recycle-bin`, Images / Classify single object)
2. **Data acquisition → Upload data**
   - `model/dataset/train` 폴더 선택 → Upload into category **Training** → Label **Infer from filename** → Upload
   - `model/dataset/test` 폴더도 같은 방법으로, category만 **Testing**
   - 자동 분할(Automatically split)은 **끄기**. 이미 원본 사진 단위로 나눠 둠
3. **Create impulse**
   - Image data: **160 × 160**, Resize mode **Fit shortest axis**
   - Processing block: **Image** / Learning block: **Transfer Learning (Images)** → Save
4. **Image**: Color depth **RGB** → Save parameters → **Generate features**
5. **Transfer learning**
   - Model: **MobileNetV2 160x160 0.35** (먼저) → 결과가 아쉬우면 0.5로 비교
   - Training cycles **40**, Learning rate **0.0005**, **Data augmentation 켜기**
   - Start training → Confusion matrix 확인
6. **Model testing → Classify all**: 테스트셋 정확도 확인 (이 숫자를 발표에 사용)
7. **Deployment → Arduino library**, **Quantized (int8)**, EON Compiler 켜기 → Build → zip 다운로드

시간 여유가 있으면 **EON Tuner**로 입력 크기·모델 조합을 자동 탐색해 보세요.

## 3. 정확도를 최대로 끌어올리는 포인트

1. **상자 안 촬영 환경을 TrashNet과 똑같이 만들기** (가장 중요)
   - 바닥을 **흰색 무광 보드**로 (TrashNet이 흰 포스터보드 위에서 촬영됨)
   - 위/앞에서 고르게 비추는 **흰색 LED**, 그림자 최소화
   - 카메라는 물체 전체가 화면의 50~80%를 차지하는 거리
2. **학습 입력 크기와 추론 입력 크기를 같게** — 카메라 QVGA(4:3) → 학습도 4:3 원본에서 Fit shortest axis
3. **신뢰도 0.70 미만은 일반 칸 + 확인 필요** (`config.h` CONF_THRESHOLD). 대시보드 오분류 신고로 약점 파악
4. 버전 기록: `config.h` 의 `MODEL_VERSION` 을 `v1-trashnet-160` 처럼 바꿔 대시보드에서 비교

## 4. 알려진 한계

- TrashNet은 미국 제품 위주라 한국 캔·페트 라벨과 모양이 다를 수 있음
- `plastic`에는 페트병 외 플라스틱 용기도 섞여 있음
- `general`의 2/3가 유리병이라 과자 봉지 같은 비닐류는 약할 수 있음 → 낮은 신뢰도로 일반 칸에 가는 경우가 많음
