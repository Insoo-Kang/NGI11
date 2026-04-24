"""
Late Fusion (Process 2 + PXRD 50 → MLP → NGI 18 targets)
검증: K-Fold Cross Validation
"""

import os
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. 설정값 (필요에 따라 수정)
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 수치형 공정 파라미터 컬럼
PROCESS_COLS = ["Blending_Time", "LPM"]

# 출력 컬럼 — Random Forest 모델의 Target_Columns와 동일
NGI_COLS = [
    "IP (mg)", "PRE (mg)",
    "Stage 1 (mg)", "Stage 2 (mg)", "Stage 3 (mg)", "Stage 4 (mg)",
    "Stage 5 (mg)", "Stage 6 (mg)", "Stage 7 (mg)", "Stage 8 (mg)",
    "Remaining (mg)", "Mouth (mg)", "PTP/Vial (mg)",
    "ED (mg)", "FPD (mg)", "FPF (%)", "RD (mg)", "SUM_ALL (mg)",
]

OUTPUT_DIM = len(NGI_COLS)  # 18

# PXRD 이외의 컬럼 집합 — CSV에서 PXRD 컬럼을 자동 감지하는 데 사용
# Random Forest의 drop(columns=Target_Columns).drop(columns='Type') 와 동일한 기준
# (헤더값이 바뀌어도 이 목록에 없는 컬럼은 전부 PXRD로 인식)
_NON_PXRD_COLS = {"Type", "Blending_Time", "LPM"} | set(NGI_COLS)

# INPUT_DIM은 load_data() 시점에 결정 (PXRD 컬럼 수에 따라 동적으로 변동)

# MLP 하이퍼파라미터
HIDDEN_DIMS  = [64, 32, 16]   # 소규모 데이터에 맞게 유지
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 300
AUGMENT_N    = 800        # 증강 후 train 샘플 수
K_FOLDS      = 5          # K-Fold 분할 수 (5 또는 10 권장)


# ─────────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    CSV 컬럼 구성:
      Type (참고용, 입력 제외), Blending_Time, LPM  : 공정 파라미터 (2열)
      _NON_PXRD_COLS 에 없는 나머지 컬럼             : PXRD intensity (N열, 자동 감지)
      IP (mg) ~ SUM_ALL (mg)                         : NGI 18개 타깃 (출력, RF와 동일)
    반환: X, y, feature_names
    """
    df = pd.read_csv(csv_path)

    # PXRD 컬럼 자동 감지: _NON_PXRD_COLS에 없는 컬럼 전부
    pxrd_cols = [c for c in df.columns if c not in _NON_PXRD_COLS]

    # 수치형 공정 파라미터
    X_proc = df[PROCESS_COLS].values.astype(np.float32)    # (N, 2)

    # PXRD intensity
    X_pxrd = df[pxrd_cols].values.astype(np.float32)       # (N, n_pxrd)

    # 입력 결합: Process 2 + PXRD n_pxrd
    X = np.concatenate([X_proc, X_pxrd], axis=1)           # (N, 2+n_pxrd)

    # NGI 출력
    y = df[NGI_COLS].values.astype(np.float32)             # (N, 18)

    feature_names = PROCESS_COLS + pxrd_cols
    return X, y, feature_names


# ─────────────────────────────────────────────
# 3. 가우시안 증강 (train fold 내부에서만 사용)
# ─────────────────────────────────────────────
def gaussian_augment(X: np.ndarray, y: np.ndarray,
                     n_samples: int = AUGMENT_N,
                     noise_std_ratio: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    각 특징의 표준편차 × noise_std_ratio 크기의 가우시안 노이즈 추가.
    원본 포함 반환.
    """
    rng = np.random.default_rng(SEED)
    std = X.std(axis=0) * noise_std_ratio + 1e-8
    idx = rng.integers(0, len(X), size=n_samples)
    noise = rng.normal(0, std, size=(n_samples, X.shape[1])).astype(np.float32)
    X_aug = np.concatenate([X, X[idx] + noise], axis=0)
    y_aug = np.concatenate([y, y[idx]], axis=0)
    return X_aug, y_aug


# ─────────────────────────────────────────────
# 4. MLP 모델 정의
# ─────────────────────────────────────────────
class MLPRegressor(nn.Module):
    """
    입력: Process 2 + PXRD n_pxrd  (load_data에서 결정)
    출력: 18차원 (IP ~ SUM_ALL)
    """
    def __init__(self, input_dim: int,
                 hidden_dims: list = HIDDEN_DIMS,
                 output_dim: int = OUTPUT_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # 선형 출력 (회귀)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 5. 단일 fold 학습
# ─────────────────────────────────────────────
def train_one_fold(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   scaler_X: StandardScaler, scaler_y: StandardScaler,
                   input_dim: int):
    """
    1. scaler fit → transform (train only)
    2. 가우시안 증강
    3. MLP 학습
    4. test 예측 후 역변환
    """
    # 스케일링 — scaler는 train에만 fit
    X_tr_sc = scaler_X.fit_transform(X_train)
    y_tr_sc = scaler_y.fit_transform(y_train)
    X_te_sc = scaler_X.transform(X_test)

    # 증강
    X_aug, y_aug = gaussian_augment(X_tr_sc.astype(np.float32),
                                    y_tr_sc.astype(np.float32))

    # DataLoader
    ds = TensorDataset(torch.from_numpy(X_aug), torch.from_numpy(y_aug))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    # 모델
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # 예측 및 역변환
    model.eval()
    with torch.no_grad():
        x_te = torch.from_numpy(X_te_sc.astype(np.float32)).to(device)
        pred_sc = model(x_te).cpu().numpy()
    if np.isnan(pred_sc).any():
        pred_sc = np.zeros_like(pred_sc)  # fallback: 스케일 공간의 평균(0)으로 대체
    pred = scaler_y.inverse_transform(pred_sc)   # (n_test, 18)
    return pred, y_test, model                   # 둘 다 원래 단위

def train_full(X: np.ndarray, y: np.ndarray) -> tuple:
    """전체 데이터로 학습 후 in-sample R2/MSE 출력 및 모델·scaler 반환."""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    pred, _, model = train_one_fold(X, y, X, y, scaler_X, scaler_y, input_dim=X.shape[1])

    mse_per_target = mean_squared_error(y, pred, multioutput="raw_values")
    r2_per_target  = r2_score(y, pred, multioutput="raw_values")
    print("\n── 전체 데이터 학습 성능 (K-Fold CV 이전, in-sample) ──")
    print(f"  {'Target':<20}  {'MSE':>12}  {'R2':>8}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}")
    for col, mse, r2 in zip(NGI_COLS, mse_per_target, r2_per_target):
        print(f"  {col:<20}  {mse:>12.4f}  {r2:>8.4f}")
    print(f"  {'전체 평균':<20}  {mse_per_target.mean():>12.4f}  {r2_per_target.mean():>8.4f}")

    return model, scaler_X, scaler_y


def save_artifacts(model: nn.Module, scaler_X: StandardScaler,
                   scaler_y: StandardScaler, input_dim: int,
                   save_dir: str = r"C:\NGI_Nintendanib\Model\MLP"):
    """모델 가중치·구조 정보·scaler를 저장."""
    os.makedirs(save_dir, exist_ok=True)

    # 모델 가중치
    model_path = os.path.join(save_dir, "mlp_model.pth")
    torch.save(model.state_dict(), model_path)

    # 모델 구조 재현에 필요한 하이퍼파라미터
    meta_path = os.path.join(save_dir, "mlp_config.pkl")
    joblib.dump({"input_dim": input_dim, "hidden_dims": HIDDEN_DIMS,
                 "output_dim": OUTPUT_DIM, "dropout": DROPOUT}, meta_path)

    # Scaler
    joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))

    print(f"\n── 모델 저장 완료: {save_dir} ──")
    print(f"  mlp_model.pth  — 모델 가중치")
    print(f"  mlp_config.pkl — 구조 메타데이터 (input_dim 등)")
    print(f"  scaler_X.pkl   — 입력 StandardScaler")
    print(f"  scaler_y.pkl   — 출력 StandardScaler")


# ─────────────────────────────────────────────
# 6. Permutation Feature Importance
# ─────────────────────────────────────────────
class _MLPSklearnWrapper:
    """permutation_importance가 요구하는 .predict() / .score() 인터페이스."""
    def __init__(self, model: nn.Module, scaler_X: StandardScaler,
                 scaler_y: StandardScaler):
        self.model    = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device   = next(model.parameters()).device

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_X.transform(X)
        x_t  = torch.from_numpy(X_sc.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_sc = self.model(x_t).cpu().numpy()
        return self.scaler_y.inverse_transform(pred_sc)

    def fit(self, X, y=None):          # permutation_importance가 요구하는 인터페이스
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(r2_score(y, self.predict(X), multioutput="uniform_average"))


def compute_feature_importance(model: nn.Module, scaler_X: StandardScaler,
                                scaler_y: StandardScaler,
                                X: np.ndarray, y: np.ndarray,
                                feature_names: list[str],
                                n_repeats: int = 30,
                                top_n: int = 20) -> pd.DataFrame:
    """
    Permutation Importance 계산.
    각 특징을 n_repeats회 무작위로 섞어 R² 감소량(mean ± std)을 측정.
    감소가 클수록 해당 특징이 예측에 중요함.
    """
    wrapper = _MLPSklearnWrapper(model, scaler_X, scaler_y)
    result = permutation_importance(wrapper, X, y,  # type: ignore[arg-type]
                                    n_repeats=n_repeats,
                                    random_state=SEED,
                                    scoring="r2")

    imp_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": result.importances_mean,  # type: ignore[union-attr]
        "Std":        result.importances_std,   # type: ignore[union-attr]
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n── Permutation Feature Importance (상위 {top_n}개, n_repeats={n_repeats}) ──")
    print(f"  {'Rank':<5}  {'Feature':<25}  {'Importance (ΔR²)':>18}  {'Std':>8}")
    print(f"  {'-'*5}  {'-'*25}  {'-'*18}  {'-'*8}")
    for i, (feat, imp, std) in enumerate(
            zip(imp_df["Feature"], imp_df["Importance"], imp_df["Std"])):
        if i >= top_n:
            break
        print(f"  {i+1:<5}  {feat:<25}  {imp:>18.4f}  {std:>8.4f}")

    return imp_df


# ─────────────────────────────────────────────
# 7. K-Fold CV 메인 루프
# ─────────────────────────────────────────────
def run_kfold(X: np.ndarray, y: np.ndarray,
              k: int = K_FOLDS) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    all_pred = np.zeros_like(y)
    all_true = np.zeros_like(y)

    for fold, (tr_idx, te_idx) in enumerate(tqdm(kf.split(X), total=k, desc=f"{k}-Fold CV 진행")):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        pred, true, _ = train_one_fold(X_train, y_train,
                                       X_test, y_test,
                                       scaler_X, scaler_y,
                                       input_dim=X.shape[1])
        all_pred[te_idx] = pred   # (n_test, 18)
        all_true[te_idx] = true   # (n_test, 18)

    return all_pred, all_true  # (N, 18)


# ─────────────────────────────────────────────
# 7. 성능 평가 출력
# ─────────────────────────────────────────────
def evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    results = []
    for i, col in enumerate(NGI_COLS):
        p, t = y_pred[:, i], y_true[:, i]
        mae  = mean_absolute_error(t, p)
        mape = np.mean(np.abs((t - p) / (t + 1e-8))) * 100
        r2   = r2_score(t, p)
        results.append({"Stage": col, "MAE": mae,
                        "MAPE(%)": mape, "R2": r2})
    df = pd.DataFrame(results)
    df.loc[len(df)] = ["전체 평균",
                       df["MAE"].mean(),
                       df["MAPE(%)"].mean(),
                       df["R2"].mean()]
    return df


# ─────────────────────────────────────────────
# 8. 실행 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "C://NGI_Nintendanib/Data/Master/Nintendanib_NGI.csv"
    print(f"데이터 로드: {csv_path}")
    X, y, feature_names = load_data(csv_path)
    n = len(X)
    print(f"  X shape: {X.shape}  →  Process 2 + PXRD {X.shape[1] - 2} = {X.shape[1]} Dimensions")
    print(f"  y shape: {y.shape}  →  NGI {OUTPUT_DIM} Targets")

    # 전체 데이터 학습 → in-sample R2/MSE 출력 + 최종 모델·scaler 반환
    final_model, scaler_X_final, scaler_y_final = train_full(X, y)

    print(f"\n{K_FOLDS}-Fold CV Start (총 {n}개 샘플)...")
    y_pred, y_true = run_kfold(X, y, k=K_FOLDS)

    print(f"\n── 예측 성능 ({K_FOLDS}-Fold CV) ──")
    result_df = evaluate(y_pred, y_true)
    print(result_df.to_string(index=False, float_format="{:.4f}".format))

    # 변수 중요도 (Permutation Importance, 전체 데이터 기준)
    imp_df = compute_feature_importance(
        final_model, scaler_X_final, scaler_y_final,
        X, y, feature_names, n_repeats=30, top_n=20,
    )

    # 모델·scaler·중요도 저장
    save_artifacts(final_model, scaler_X_final, scaler_y_final,
                   input_dim=X.shape[1])
    save_dir = r"C:\NGI_Nintendanib\Model\MLP"
    imp_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
    print("  feature_importance.csv — Permutation Importance 전체 순위")
