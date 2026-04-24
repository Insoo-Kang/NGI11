import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
import joblib
import time
from tqdm import tqdm


from pathlib import Path
Workspace_root = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

Target_Columns = [
    'IP (mg)','PRE (mg)','Stage 1 (mg)','Stage 2 (mg)',
    'Stage 3 (mg)','Stage 4 (mg)','Stage 5 (mg)',
    'Stage 6 (mg)','Stage 7 (mg)','Stage 8 (mg)',
    'Remaining (mg)','Mouth (mg)','PTP/Vial (mg)',
    'ED (mg)','FPD (mg)','FPF (%)','RD (mg)','SUM_ALL (mg)'
]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def Prediction_Data_Prepare(
    crystal_shape='',
    Blending_Time=30,
    LPM=60):
    
    # ============================
    # 1. Prepare Prediction Data
    # ============================
    """ Prepare a single-row DataFrame for NGI prediction based on the given crystal shape and process conditions."""
    _PXRD_Data =  pd.read_csv(Workspace_root/'Data'/'Master'/'Selected_PXRD_Features.csv', index_col=0)
    Crystal_Shape = crystal_shape.strip()
    Shape_Map = {str(idx).strip().lower(): idx for idx in _PXRD_Data.index}
    Matched_shape = Shape_Map.get(Crystal_Shape.lower())

    if Matched_shape is None:
        print(f"Crystal Shape '{Crystal_Shape}' not found. Available values: {list(_PXRD_Data.index)}")
        return pd.DataFrame()  # Return empty DataFrame if shape not found

    Fixed_PXRD_Values = _PXRD_Data.loc[[Matched_shape]]

    Prediction_Data_Conditions = pd.DataFrame({
        'Blending_Time': [Blending_Time],
        'LPM': [LPM]})
    
    Prediction_Inputs = pd.concat([Prediction_Data_Conditions, Fixed_PXRD_Values.reset_index(drop=True)], axis=1)

    # ============================
    # 2. Predict NGI Value
    # ============================

    # 모델 하이퍼파라미터 및 Scaler 로드 (scaler_y.pkl 대소문자 주의)
    mlp_config = joblib.load(Workspace_root / "Model" / "MLP" / "mlp_config.pkl")
    MLP_NGI_scaler_X = joblib.load(Workspace_root / "Model" / "MLP" / "scaler_X.pkl")
    MLP_NGI_scaler_y = joblib.load(Workspace_root / "Model" / "MLP" / "scaler_y.pkl")

    # 모델 구조 초기화 및 가중치(State Dict) 로드
    MLP_NGI_Model = MLPRegressor(
        input_dim=mlp_config["input_dim"],
        hidden_dims=mlp_config["hidden_dims"],
        output_dim=mlp_config["output_dim"],
        dropout=mlp_config["dropout"]
    )
    MLP_NGI_Model.load_state_dict(torch.load(Workspace_root / "Model" / "MLP" / "mlp_model.pth", map_location='cpu', weights_only=True))
    MLP_NGI_Model.eval()

    Scaled_X = MLP_NGI_scaler_X.transform(Prediction_Inputs)

    # 텐서 변환 및 예측
    with torch.no_grad():
        x_tensor = torch.from_numpy(Scaled_X.astype(np.float32))
        Prediction_Outputs_Scaled = MLP_NGI_Model(x_tensor).numpy()

    # 모델의 출력값은 StandardScaler로 정규화된 값이므로 역변환(Inverse Transform) 적용
    Prediction_Outputs = MLP_NGI_scaler_y.inverse_transform(Prediction_Outputs_Scaled)[0]

    Prediction_Results = pd.DataFrame([Prediction_Outputs], columns=Target_Columns)
    
    return Prediction_Results

if __name__ == "__main__":
    print("Input Crystal Shape (Ex. NE_Rod_1):")
    crystal_shape_input = input('').strip()
    if crystal_shape_input == "":
        crystal_shape_input = "NE_Rod_1"
    print("Input Blending Time (min, default 30):")
    blending_time_input = input().strip()
    if blending_time_input == "":
        blending_time_input = 30
    print("Input LPM (default 60):")
    lpm_input = input().strip()
    if lpm_input == "":
        lpm_input = 60
    Prediction_Results = Prediction_Data_Prepare(
        crystal_shape=crystal_shape_input,
        Blending_Time=int(blending_time_input),
        LPM=int(lpm_input)
    )
    print("Predicted NGI Values:")
    print(Prediction_Results.to_string(index=False))