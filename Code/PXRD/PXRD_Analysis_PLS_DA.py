import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from pathlib import Path
Workspace_root = Path(__file__).resolve().parents[2]
# =================
# 새로운 PXRD 측정 데이터를 추가 후 분석하고자 한다면
# Nintendanib_Morphology_PXRD.csv 파일에 표준화한 측정값을 넣고
# 첫 행(결정형)에는 "NE_결정형_번호" (Example. NE_Raw_3)를 차례대로 추가한 이후 code를 실행하면 됩니다.
# =================

# =================
# Permission Error가 terminal에 뜨는 경우, CSV 파일을 엑설에서 저장 후 닫았는지 확인하세요! 
# =================

# =================
# 1. Define VIP(Variable Importance in Projection) Score Function
# =================

def Calculate_VIP(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p * np.dot(s, weight) / total_s)
    return vips

# =================
# 2. Prepare PXRD Data
# =================
PXRD = pd.read_csv(Workspace_root / "Data" /"Master"/ "Nintendanib_Morphology_PXRD.csv")

# Column 뒤 공백 제거
PXRD.columns = PXRD.columns.str.strip()

# Transpose (X축: Angle, Y축: 결정형)
X_PXRD = PXRD.set_index('Angle').T

# Extract Crystal Group From Data Frame & Set Target
Y_Labels = []
for Crystal_Name in X_PXRD.index:
    name_lower = Crystal_Name.lower()
    if 'raw' in name_lower: Y_Labels.append('Form_Raw')
    elif 'needle' in name_lower: Y_Labels.append('Form_Needle')
    elif 'rod' in name_lower: Y_Labels.append('Form_Rod')
    elif 'small' in name_lower or 'square' in name_lower:Y_Labels.append('Form_Small_Square')
    else: Y_Labels.append('Unknown')

# One Hot Encoding For Target Values (Crystal Type)
Y_Forms = pd.get_dummies(Y_Labels) # type: ignore

print("==============================================")
print(f"Total Samples: {len(Y_Labels)}")
print("==============================================")
print("Distribution of Sample by Group:\n", pd.Series(Y_Labels).value_counts())
print("==============================================")
print("=== Transposed PXRD Data ===")
print(f"Number of Samples: {X_PXRD.shape[0]}, 2 theta angles: {X_PXRD.shape[1]}")
print("==============================================")
print(X_PXRD.head())
print("==============================================")

# =================
# 3. Set PLS-DA model target
# =================

# ==================
#4. PLS-DA Model Training
# ==================
# Sample 수가 8개(결정형 8종), n_components는 최대 8-1=7개, 
# n_components: 모델 내부적으로 만드는 주성분(잠재 변수)의 개수
PLS = PLSRegression(n_components = 5)
PLS.fit(X_PXRD, Y_Forms)

#R2: (Model Fit): 현재 데이터에 대한 피팅 정확도
Y_Pred = PLS.predict(X_PXRD)
R2 = r2_score(Y_Forms, Y_Pred)

#Q2: (Predictive Perfomance): 교차 검증 (LOOCV)를 통한 예측 정확도.
loo = LeaveOneOut()
Y_CV_Pred = cross_val_predict(PLS, X_PXRD, Y_Forms, cv=loo)
Q2 = r2_score(Y_Forms, Y_CV_Pred)

print(f"========= PLS-DA Validation Metrics =========")
print(f"R2 (Coefficient of determination): {R2:.4f}") # 설명력
print("==============================================")
print(f"Q2 (Predictive Capability): {Q2:.4f}") # 예측력
print("==============================================")


# ==================
#4. Calculate VIP Score
# ==================
# PLS-DA (Partial Least Squares-Discriminant Analysis)
# VIP Score: 
VIP_Scores = Calculate_VIP(PLS)
VIP_df = pd.DataFrame({'2_Theta': X_PXRD.columns, 'VIP_Score': VIP_Scores})

# Exract Top 10 VIP Scores (VIP Score >= 1.0 이어야 유의미함!)
Top_features = VIP_df[VIP_df["VIP_Score"]>=1.0].sort_values(by='VIP_Score', ascending=False).head(10)
Selected_Columns = Top_features['2_Theta'].tolist()

print("\n=== Top 10 2 theta points to describe morphology ===")
print(Top_features.to_string(index=False))

PXRD_Selected = X_PXRD[Selected_Columns]

# ==================
# 5. Create Minimized Data Frame
# ==================

Output_File = Workspace_root / "Data" / "Master" / "Selected_PXRD_features.csv"

# Index(2_theta)에 'Angle'이라는 이름을 붙여서 저장
PXRD_Selected.index.name = 'Angle'
PXRD_Selected.to_csv(Output_File)

print(f"Saved as '{Output_File}'!")