import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import time
from pathlib import Path
Workspace_root = Path(__file__).resolve().parents[2]

random_state = 42

# ============================================
# 1. Import Data
# ============================================
NGI = pd.read_csv(Workspace_root/'Data'/'Master'/'Nintendanib_NGI.csv')

# Target Variable Setting
Target_Columns = [
    'IP (mg)','PRE (mg)','Stage 1 (mg)','Stage 2 (mg)',
    'Stage 3 (mg)','Stage 4 (mg)','Stage 5 (mg)',
    'Stage 6 (mg)','Stage 7 (mg)','Stage 8 (mg)',
    'Remaining (mg)','Mouth (mg)','PTP/Vial (mg)',
    'ED (mg)','FPD (mg)','FPF (%)','RD (mg)','SUM_ALL (mg)'
]

X = NGI.drop(columns=Target_Columns).drop(columns='Type')
Y = NGI[Target_Columns]
print("=====================================================")
print(X.head())
print("=====================================================")
print(X.describe())
print("=====================================================")
print(Y.head())
print("=====================================================")
print(Y.describe())
print("=====================================================")

# ============================================
# 2. Multi Output Random Forest Model Training
# ============================================

RF_NGI = RandomForestRegressor(
    n_estimators = 0,
    max_depth=5,
    warm_start=True,
    n_jobs=-1,
    random_state=random_state
)

Total_Trees = 1000
Step = 50
Start_Time = time.time()

# Print Stopwatch using Relay Method
print(f"\n[START] Training (Total {Total_Trees}) Trees")
for i in range(Step, Total_Trees + 1, Step):
    # Increase tree 50 per trial.
    RF_NGI.n_estimators = i  # type: ignore
    
    # 훈련! (warm_start 덕분에 처음부터 다시 만들지 않고 추가된 50개만 새로 만듭니다)
    RF_NGI.fit(X, Y)         
    
    # 시간 계산
    elapsed = time.time() - Start_Time
    avg_time_per_tree = elapsed / i
    eta = avg_time_per_tree * (Total_Trees - i)
    
    progress_pct = (i / Total_Trees) * 100

    # \r을 사용하여 콘솔창 한 줄에 계속 덮어쓰기 (애니메이션 효과)
    sys.stdout.write(f"\r[TREES] Created: [{i}/{Total_Trees} ({progress_pct:.1f}%)] "
                     f"[TIME] Elapsed: {elapsed:.1f} sec | "
                     f"[ETA] Completion: {eta:.2f} sec")
    sys.stdout.flush()

total_elapsed = time.time() - Start_Time
print(f"\n[DONE] Model Creating Complete! (Elapsed Time: {total_elapsed:.1f} sec)\n")

# ==========================================
# 3. Model Validation
# Validation of Total Random Forest Model
# LOOCV (Leave One Out Cross-Validation)
# ==========================================
loo= LeaveOneOut()
Splits = list(loo.split(X))
Total_Splits = len(Splits)

# Use a separate CV model without warm_start.
# Re-fitting the same warm_start model with unchanged n_estimators
# triggers: "Warm-start fitting without increasing n_estimators..."
RF_NGI_CV = RandomForestRegressor(
    n_estimators=RF_NGI.n_estimators, # type: ignore
    max_depth=RF_NGI.max_depth, # type: ignore
    warm_start=False,
    n_jobs=-1,
    random_state=random_state
)

Y_CV_Pred = np.zeros_like(Y.to_numpy(), dtype=float)

# Validation Loop(tqdm progress bar)
print("\n[START] LOOCV Validation")
for train_index, test_index in tqdm(Splits, total=Total_Splits, desc="LOOCV Progress"):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    RF_NGI_CV.fit(X_train, Y_train)
    Y_CV_Pred[test_index] = RF_NGI_CV.predict(X_test)

R_square_Total = r2_score(Y, Y_CV_Pred)
MAE_Total = mean_absolute_error(Y, Y_CV_Pred)
RMSE_Total = np.sqrt(mean_squared_error(Y, Y_CV_Pred))

print("==== Accuracy of Total Random Forest Model (R² & MAE & MSE)====")
print(f"[R²: {R_square_Total:5.2f}  |  MAE (Mean Absolute Error):{MAE_Total:5.2f} |  RMSE (Root Mean Squared Error): {RMSE_Total:5.2f}]")

# ==========================================
# Check Accuracy of Each Variables
# ==========================================
print("==== Accuracy of Each Variables (R² & MAE & RMSE)====")
for i, col in enumerate(Target_Columns):
 R2=r2_score(Y.iloc[:, i], Y_CV_Pred[:, i])
 MAE = mean_absolute_error(Y.iloc[:, i], Y_CV_Pred[:, i])
 RMSE = np.sqrt(mean_squared_error(Y.iloc[:, i], Y_CV_Pred[:, i]))
 print(f"[{col:^8}] R²: {R2:5.2f}  |  MAE: {MAE:5.2f} |  RMSE: {RMSE:5.2f}")
print("=====================================================")

# ==========================================
# Check Feature Importance (Standard: FPF (%))
# ==========================================
# Random Forest Targets FPF(%) only
Y_FPF = Y['FPF (%)']

# Training Model
RF_FPF = RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=5)

Start_Time = time.time()
RF_FPF.fit(X, Y_FPF)
End_Time = time.time()
print(f"Time for Validating FPF(%) (sec): {End_Time - Start_Time:.4f}")

# FPF 기준 feature importance
Importance_FPF = RF_FPF.feature_importances_
Importance_FPF_DF = pd.DataFrame({
    'Feature': X.columns,
    'Importance': Importance_FPF
}).sort_values(by='Importance', ascending=False)
print("=====================================================")
print("==== Top 5 Important Feature Predicting FPF(%) ====")
print(Importance_FPF_DF.head(5).to_string(index=False))
print("=====================================================")


# ==========================================
# 5. Save Model as File
# ==========================================
import joblib
# Model Save Folder & File Name
model_save_path = Workspace_root/'Model'/'NGI_Random_Forest_Model.pkl'

# Save Random Forest NGI Model
joblib.dump(RF_NGI, model_save_path)
print("=====================================================")
print(f"Model is saved at '{model_save_path}'")
print("=====================================================")
