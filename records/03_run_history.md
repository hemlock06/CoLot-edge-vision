# 03 — 실행 기록

## 환경
- GPU: CUDA 사용(torch 2.6.0+cu124, `cuda.is_available()=True`)
- 패키지: ultralytics 8.4 · easyocr 1.7.2 · onnx 1.22 · onnxruntime 1.27 · onnxslim · scikit-learn 1.9 · opencv 4.13
- 한글 폰트: `C:\Windows\Fonts\malgun*.ttf` (번호판 렌더 + matplotlib)

## 빌드 순서
1. `scripts/eval_adaptive.py` — ③ 분류기 학습(`data/adaptive_clf.joblib`) + 전력/커버리지.
2. `scripts/build_plate.py` — 합성 YOLO 데이터셋 → YOLOv8n 학습(`best.pt`) →
   ONNX export → 정적 INT8(`best_int8.onnx`) → 벤치 → OCR. (학습된 `best.pt`는 재사용.)
3. `scripts/eval_anomaly.py` — ② 시뮬·룰·IForest 평가.
4. `scripts/eval_weather_ocr.py` · `make_gallery.py` · `make_viz2.py` — 악천후·갤러리·시각화.

## 해결한 문제
- **콘솔 cp949 깨짐**: EasyOCR 다운로드 진행바(█)·한글 출력 → `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.
- **정적 INT8 InvalidGraph**: opset 12의 DequantizeLinear가 per-channel `axis` 미지원
  → export opset **13**으로 상향.
- **동적 양자화 지연 역행(198ms)** → 정적 양자화로 전환(→ 01 D2).
- **메모리 OOM / 페이징 부족(WinError 1455)**: ultralytics 학습의 DataLoader 워커가
  누수되어 python 프로세스 26개·8.8GB 점유, torch-CUDA+easyocr+onnxruntime 동시 로드 시
  페이징 부족으로 cudnn 로드 실패. 해결:
  - 학습 `workers=0, cache=False`로 워커 누수 차단.
  - 평가에서 GPU 모델 사용 후 `del + torch.cuda.empty_cache()` → 그 다음 onnx/easyocr 로드.
  - EasyOCR 1개 인스턴스를 클라우드/엣지 단계가 공유(`PlateReader(ocr_reader=…)`).
  - 고아 프로세스 정리: `Get-Process python | Stop-Process -Force` → RAM 8.7GB 회수.

## 재현
```bash
PYTHONUTF8=1 python scripts/eval_adaptive.py
PYTHONUTF8=1 python scripts/build_plate.py
PYTHONUTF8=1 python scripts/eval_anomaly.py
pytest -q     # 26 passed
```
