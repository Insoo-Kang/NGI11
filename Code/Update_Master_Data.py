import pandas as pd
from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
WORKSPACE  = Path(__file__).resolve().parents[1]   # c:\NGI_Nintendanib
MASTER_DIR = WORKSPACE / "Data" / "Master"

PXRD_PATH   = MASTER_DIR / "Selected_PXRD_features.csv"   # 선정된 PXRD 피크 특성값
NGI_PATH    = MASTER_DIR / "Nintendanib_NGI.csv"           # NGI 실험 결과 (원본)
SEM_PATH    = MASTER_DIR    / "Crystal_Measurements.csv"      # SEM 결정 측정값 (결정형 4종)
OUTPUT_PATH = MASTER_DIR / "Nintendanib_NGI.csv"    # 최종 병합 저장 경로


# ============================================================
# 사용법 안내
# ============================================================
# 이 스크립트는 세 가지 데이터를 결합하여 모델 학습용 데이터를 생성합니다.
#
# [실행 전 준비사항]
# 1. PXRD 피크가 변경된 경우 → PXRD_Analysis_PLS_DA.py 먼저 실행해 Selected_PXRD_features.csv 갱신
# 2. 새 실험 결과 추가 시   → Nintendanib_NGI.csv 에 PXRD/SEM 제외 항목 직접 입력
#    - Type        : 결정형 이름 (예: NE_Raw_1, NE_Needle_2 …)
#    - Blending_Time, LPM, Stage별 약물량 등
#    - ※ 빈칸(NaN) 없이 완전히 채워야 합니다
#    - ※ 동일 조건의 반복 실험은 평균이 아닌 회차별 행으로 각각 입력하세요
#
# [실행 결과]
# - PXRD 피크값과 SEM 결정 측정값이 Type 기준으로 자동 매핑되어 저장됩니다


# ============================================================
# 1. 데이터 로드
# ============================================================
pxrd_df   = pd.read_csv(PXRD_PATH)
master_df = pd.read_csv(NGI_PATH)
sem_df    = pd.read_csv(SEM_PATH)

print("=" * 60)
print(f"[로드 완료]")
print(f"  PXRD features : {pxrd_df.shape}  ({PXRD_PATH.name})")
print(f"  NGI master    : {master_df.shape}  ({NGI_PATH.name})")
print(f"  SEM 측정값    : {sem_df.shape}  ({SEM_PATH.name})")
print("=" * 60)


# ============================================================
# 2. PXRD 피크값 매핑
#    Type(예: NE_Raw_1)을 기준으로 Selected_PXRD_features 의 피크 강도값을 매핑
#
#    Selected_PXRD_features.csv 구조:
#      행  : 결정형 타입 (Angle 컬럼에 저장)
#      열  : 2θ 각도값 (예: 19.9008, 15.8019 …)
# ============================================================
PXRD_TYPE_COL   = "Angle"   # PXRD 파일에서 타입 식별자 역할을 하는 컬럼
MASTER_TYPE_COL = "Type"    # NGI 파일의 결정형 타입 컬럼

pxrd_peak_cols = [c for c in pxrd_df.columns if c != PXRD_TYPE_COL]

# Type → 피크 강도 조회 테이블 생성 (인덱스: NE_Raw_1, NE_Needle_2 …)
pxrd_lookup = pxrd_df.set_index(PXRD_TYPE_COL)[pxrd_peak_cols]

# 신규 컬럼 여부 알림
new_pxrd_cols = [c for c in pxrd_peak_cols if c not in master_df.columns]
if new_pxrd_cols:
    print(f"[PXRD] 신규 컬럼 추가: {new_pxrd_cols}")

# 매핑 적용
for col in pxrd_peak_cols:
    master_df[col] = master_df[MASTER_TYPE_COL].map(pxrd_lookup[col])

# NaN 경고 출력
pxrd_nan = master_df[pxrd_peak_cols].isna().sum()
if pxrd_nan.sum() > 0:
    print("[WARNING] PXRD 매핑 실패 - NaN 발생 컬럼:")
    print(pxrd_nan[pxrd_nan > 0].to_string())
else:
    print(f"[PXRD] 매핑 완료 — {len(pxrd_peak_cols)}개 피크 컬럼, NaN 없음")


# ============================================================
# 3. SEM 결정 측정값 매핑
#    SEM 파일은 결정형 4종(NE_Raw, NE_Needle, NE_Rod, NE_Small_Square)만 존재
#    NGI 파일의 Type(NE_Raw_1, NE_Raw_2 등)에서 접미사(_1, _2)를 제거해 베이스 타입으로 매핑
#
#    예시: NE_Raw_1  → NE_Raw  → SEM 측정값 조회
#          NE_Needle_2 → NE_Needle → SEM 측정값 조회
# ============================================================
SEM_ID_COL = "sample_id"   # SEM 파일의 결정형 타입 컬럼

sem_feature_cols = [c for c in sem_df.columns if c != SEM_ID_COL]

# 베이스 타입 추출: 마지막 '_숫자' 접미사 제거
#   NE_Raw_1       → NE_Raw
#   NE_Small_Square_2 → NE_Small_Square
base_type_series = master_df[MASTER_TYPE_COL].str.rsplit("_", n=1).str[0]

# base_type → SEM 값 조회 테이블 생성 (인덱스: NE_Raw, NE_Needle …)
sem_lookup = sem_df.set_index(SEM_ID_COL)[sem_feature_cols]

# 신규 컬럼 여부 알림
new_sem_cols = [c for c in sem_feature_cols if c not in master_df.columns]
if new_sem_cols:
    print(f"[SEM] 신규 컬럼 추가: {len(new_sem_cols)}개")

# 매핑 적용
for col in sem_feature_cols:
    master_df[col] = base_type_series.map(sem_lookup[col])

# NaN 경고 출력
sem_nan = master_df[sem_feature_cols].isna().sum()
if sem_nan.sum() > 0:
    print("[WARNING] SEM 매핑 실패 - NaN 발생 컬럼:")
    print(sem_nan[sem_nan > 0].to_string())
else:
    print(f"[SEM] 매핑 완료 — {len(sem_feature_cols)}개 특성 컬럼, NaN 없음")


# ============================================================
# 4. 저장
# ============================================================
master_df = master_df.reset_index(drop=True)
master_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("=" * 60)
print(f"[저장 완료] {OUTPUT_PATH}")
print(f"최종 데이터 크기: {master_df.shape[0]}행 × {master_df.shape[1]}열")
print("=" * 60)
print(master_df.head(3).to_string())
