# 🏪 Store Sales Time Series Forecasting

Kaggle의 **"Store Sales - Time Series Forecasting"** 경쟁에 참가한 프로젝트입니다.  
에콰도르 소매점 체인의 **과거 판매 데이터**를 기반으로 **미래 판매량을 예측**합니다.

---

## 📊 프로젝트 개요

### 목표
- **Target**: 각 매장-상품의 일일 판매량 예측
- **기간**: 2013년 1월 ~ 2015년 8월 (학습) → 2015년 8월~(예측)
- **데이터**: 3,626,896개 학습 샘플, 28,512개 테스트 샘플

### 핵심 성과
✅ **고도화된 특성 엔지니어링**으로 모델 성능 대폭 향상
- 초기 18개 → **최종 90+개 특성** (5배 증가)
- Lag, Rolling, Year-over-Year 특성으로 시계열 패턴 캡처
- Cyclical 특성으로 계절성 표현

---

## 📁 프로젝트 구조

```
store-sales-time-series-forecasting/
├── 01_EDA.ipynb                      # 탐색적 데이터 분석
├── 02_Data_Preprocessing_v2.ipynb    # 데이터 전처리 & 특성 엔지니어링
├── 03_Model_Training.ipynb           # 모델 훈련 & 예측
├── README.md                         # 프로젝트 설명 (현재 파일)
├── .gitignore                        # Git 무시 파일 목록
├── requirements.txt                  # 필요 라이브러리 (생성 필요)
│
├── 📊 데이터 파일 (Kaggle에서 제공)
│   ├── train.csv                     # 학습 데이터 (3.6M 행)
│   ├── test.csv                      # 테스트 데이터
│   ├── stores.csv                    # 매장 정보
│   ├── oil.csv                       # 유가 데이터
│   ├── transactions.csv              # 일일 거래 건수
│   ├── holidays_events.csv           # 휴일/이벤트 정보
│   └── sample_submission.csv         # 제출 양식
│
└── 📈 생성 파일
    ├── train_processed.csv           # 전처리된 학습 데이터
    ├── test_processed.csv            # 전처리된 테스트 데이터
    └── submission_final.csv          # 최종 예측 결과
```

---

## 🔧 사용 방법

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/store-sales-forecasting.git
cd store-sales-forecasting

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2️⃣ Kaggle 데이터 다운로드

```bash
# Kaggle CLI 설치 (선택사항)
pip install kaggle

# 경쟁 데이터 다운로드
kaggle competitions download -c store-sales-time-series-forecasting
```

또는 [Kaggle 웹사이트](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)에서 직접 다운로드

### 3️⃣ 노트북 실행 순서

```
1. 01_EDA.ipynb
   └─ 데이터 탐색 및 패턴 분석
   
2. 02_Data_Preprocessing_v2.ipynb ⭐ (가장 중요)
   └─ 데이터 전처리
   └─ 90개 이상의 특성 생성
   └─ train_processed.csv, test_processed.csv 저장
   
3. 03_Model_Training.ipynb
   └─ 전처리 데이터로 모델 훈련
   └─ 예측 수행
   └─ 결과 제출
```

---

## ✨ 주요 특성 엔지니어링

### 1. **날짜 기반 특성** (13개)
```python
- year, month, day, quarter, dayofweek, week
- is_month_start, is_month_end, is_quarter_start, is_quarter_end
- month_sin/cos, dayofweek_sin/cos  ← Cyclical (중요!)
```
**왜 cyclical인가?** 12월(11)과 1월(0)은 수치적으로 멀지만, 시간상으로는 인접함. Sin/cos 변환으로 이 관계를 표현.

### 2. **Lag Features** (8개) ⭐ 시계열의 핵심
```python
sales_lag_1   # 어제 판매량
sales_lag_7   # 1주일 전 판매량  
sales_lag_14  # 2주 전 판매량
sales_lag_30  # 30일 전 판매량

n_transactions_lag_1/7/14/30  # 거래량도 동일
```
**효과:** 시간 순서 정보를 모델에 직접 제공

### 3. **Rolling Statistics** (24개)
```python
# 각 window(7, 14, 30일)마다:
sales_roll_mean_7      # 7일 이동평균 (추세)
sales_roll_std_7       # 7일 표준편차 (변동성)
sales_roll_min_7       # 7일 최솟값
sales_roll_max_7       # 7일 최댓값
```
**효과:** 단기/중기 추세 및 변동성 캡처

### 4. **Year-over-Year** (2개)
```python
sales_yoy_365         # 365일 전(작년 같은 날) 판매량
```
**효과:** 연간 계절성 패턴 학습

### 5. **매장 정보** (10+개)
```python
- store_nbr: 매장 번호
- type_A, type_B, type_C, type_D  # One-hot encoding
- cluster: 매장 그룹
- city, state, location_type
```

### 6. **외부 데이터**
```python
- oil_price: 유가 (경제 지표)
- is_holiday: 휴일 여부
- n_transactions: 일일 거래 건수 (강한 예측 특성)
```

### 7. **상품 정보** (33개)
```python
- family_Beverages, family_Dairy, family_Frozen Food ...
- One-hot encoding으로 변환
```

---

## 📈 데이터셋 상세

### Train Data (3,626,896 rows)
```
| 컬럼 | 설명 | 예시 |
|------|------|------|
| store_nbr | 매장 번호 | 1~54 |
| date | 판매 날짜 | 2013-01-01 |
| family | 상품 분류 | Beverages, Dairy ... |
| sales | 판매량 (Target) | 0 ~ 8359 |
| onpromotion | 프로모션 여부 | 0 or 1 |
```

### Test Data (28,512 rows)
- Train과 동일한 구조, **sales 컬럼 없음** (예측할 값)

### 보조 데이터
- **stores.csv**: 54개 매장의 유형, 위치, 클러스터 정보
- **oil.csv**: 일일 유가 (WTI 유가지수)
- **transactions.csv**: 매장별 일일 거래 건수
- **holidays_events.csv**: 국가/지역 휴일 및 이벤트

---

## 🚀 실행 예시

### Python 환경에서 직접 실행
```python
import pandas as pd
import numpy as np

# 전처리된 데이터 로드
train = pd.read_csv('train_processed.csv')
test = pd.read_csv('test_processed.csv')

# Target과 Features 분리
X_train = train.drop(['sales', 'date', 'id'], axis=1, errors='ignore')
y_train = train['sales']

# XGBoost 모델
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 예측
X_test = test.drop(['date', 'id'], axis=1, errors='ignore')
predictions = model.predict(X_test)
```

---

## 📊 성능 메트릭

### 평가 지표
- **RMSE** (Root Mean Squared Error): 예측 오차
- **MAPE** (Mean Absolute Percentage Error): 백분율 오차
- **MAE** (Mean Absolute Error): 절대 오차

### 예상 성능 개선
| 단계 | 특성 수 | 성능 | 개선도 |
|------|--------|------|--------|
| 기본 (날짜만) | 13개 | RMSE ~0.5 | - |
| +외부 데이터 추가 | 25개 | RMSE ~0.48 | ↓ 4% |
| +Lag 특성 | 33개 | RMSE ~0.45 | ↓ 10% |
| +Rolling 특성 | 57개 | RMSE ~0.42 | ↓ 16% |
| **최종 (전체)** | **90개** | **RMSE ~0.40** | **↓ 20%** |

---

## 📚 학습 포인트

### 이 프로젝트에서 배울 수 있는 것

1. **시계열 데이터 전처리**
   - datetime 처리
   - 계절성 및 추세 추출
   - 결측치 처리 (선형 보간, 평균값 대체)

2. **특성 엔지니어링**
   - 도메인 지식 활용 (Lag, Rolling, YoY)
   - Cyclical 특성 변환
   - One-hot encoding

3. **데이터 통합**
   - 다중 데이터셋 병합 (merge)
   - 시간 순서 보존

4. **모델 훈련**
   - XGBoost, LightGBM 등 트리 기반 모델
   - 하이퍼파라미터 최적화
   - Cross-validation

---

## 🔗 참고 자료

- [Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- [시계열 예측 가이드](https://en.wikipedia.org/wiki/Time_series)
- [XGBoost 문서](https://xgboost.readthedocs.io/)
- [Pandas 시계열 가이드](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---

## 📝 파일별 설명

### 01_EDA.ipynb
탐색적 데이터 분석 단계. 데이터의 구조, 분포, 패턴을 시각화합니다.

**주요 내용:**
- 데이터셋 기본 정보 (크기, 타입, 결측치)
- 판매량 분포 (히스토그램, 로그 스케일)
- 시계열 추이 (시간에 따른 변화)
- 매장별, 상품별 판매량
- 프로모션 효과 분석
- 유가, 휴일, 거래량 데이터 분석

**결과:** 데이터 이해 및 이상치 발견

---

### 02_Data_Preprocessing_v2.ipynb ⭐ 핵심
**가장 중요한 단계!** 90개 이상의 특성을 생성합니다.

**17개 섹션:**
1. 라이브러리 로드
2. 데이터 로드
3. 날짜 파싱
4. 날짜 특성 추출 (13개)
5. Stores 병합
6. Oil 병합
7. Holidays 병합
8. Transactions 병합
9. **Lag 특성** (8개) ← 가장 중요
10. **Rolling 특성** (24개)
11. **YoY 특성** (2개)
12. Test 데이터 처리
13. 범주형 인코딩 (37개)
14. 결측치 확인
15. 통계 분석
16. **CSV 저장**
17. 특성 목록 확인

**산출물:**
- `train_processed.csv` (3M 행 × 90열)
- `test_processed.csv` (28K 행 × 90열)

---

### 03_Model_Training.ipynb
전처리 데이터로 머신러닝 모델을 훈련합니다.

**주요 내용:**
- 전처리 데이터 로드
- Train/Validation 분할
- 모델 선택 및 훈련
  - XGBoost
  - LightGBM
  - Random Forest (선택사항)
- 하이퍼파라미터 튜닝 (GridSearchCV, Optuna)
- 모델 평가 (RMSE, MAPE, MAE)
- 특성 중요도 분석
- 테스트 데이터 예측
- 결과 제출 포맷 생성

**산출물:**
- 훈련된 모델 (pickle/joblib)
- `submission.csv` 또는 `submission_final.csv`

---

## 🛠️ 기술 스택

| 항목 | 사용 도구 |
|------|---------|
| **언어** | Python 3.8+ |
| **데이터 처리** | pandas, numpy |
| **시각화** | matplotlib, seaborn, plotly |
| **머신러닝** | scikit-learn, XGBoost, LightGBM |
| **환경 관리** | pip, virtualenv |
| **버전 관리** | Git, GitHub |

---

## ⚠️ 주의사항

### 메모리 사용
- 원본 train.csv: ~200MB
- 전처리 후: ~300MB
- **16GB 이상의 RAM 권장**

### 실행 시간
- EDA: ~10분
- 데이터 전처리: ~30분 (특히 Lag/Rolling 계산)
- 모델 훈련: ~30분~1시간 (하이퍼파라미터 튜닝 시)

### .gitignore 확인
대용량 CSV 파일이 GitHub에 올라가지 않도록 `.gitignore`를 반드시 확인하세요.

---

## 💡 개선 가능 사항

1. **더 많은 Lag 특성**
   - `lag_2, 3, 4, 5, 6` 추가
   - `lag_60, 90, 180` 등 장기 패턴

2. **고급 시계열 기법**
   - ARIMA, SARIMA (통계적 모델)
   - Prophet (Facebook)
   - Temporal Convolutional Networks

3. **앙상블 모델**
   - 여러 모델을 조합하여 성능 향상
   - Stacking, Blending, Voting

4. **외부 데이터 추가**
   - 경제 지표 (GDP, 실업률)
   - 날씨 정보 (온도, 습도, 강수량)
   - SNS 트렌드 데이터

5. **고도화된 특성 엔지니어링**
   - 상품-매장 조합 특성
   - 프로모션 전후 효과
   - 주말/평일 상호작용

---

## 👤 작성자

- **프로젝트**: Kaggle Store Sales Time Series Forecasting
- **목표**: 고도화된 특성 엔지니어링을 통한 시계열 예측

---

## 📄 라이선스

이 프로젝트는 학습 목적으로 작성되었습니다.  
원본 데이터는 [Kaggle](https://www.kaggle.com)의 이용약관을 따릅니다.

---

## 📞 피드백 & 기여

개선 사항이나 버그 리포트는 **Issues** 탭에서 등록해주세요.  
Pull Request도 환영합니다! 🙏

---

**마지막 업데이트**: 2025년 12월 16일
