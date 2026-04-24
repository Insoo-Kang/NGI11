import pandas as pd
import numpy as np
from pathlib import Path
import joblib

Workspace_root = Path(__file__).resolve().parents[2]

RF_NGI_Model = joblib.load(Workspace_root / "Model" / "NGI_Random_Forest_Model.pkl")
Target_cols = [
        "IP (mg)","PRE (mg)","Stage 1 (mg)","Stage 2 (mg)","Stage 3 (mg)","Stage 4 (mg)",
        "Stage 5 (mg)","Stage 6 (mg)","Stage 7 (mg)","Stage 8 (mg)",
        "Remaining (mg)","Mouth (mg)","PTP/Vial (mg)","ED (mg)","FPD (mg)","FPF (%)","RD (mg)","SUM_ALL (mg)",
    ]

PXRD_Data = pd.read_csv(Workspace_root/'Data'/'Master'/'Selected_PXRD_Features.csv', index_col=0)
SEM_Data = pd.read_csv(Workspace_root/'Data'/'Master'/'Crystal_Measurements.csv', index_col=0)


# ============================
# 1. Prepare Prediction Data
# ============================

def Prediction_Data_Prepare(
    crystal_shape='',
    Blending_Time=30,
    LPM=60):

    Crystal_Shape = crystal_shape.strip()
    Shape_Map = {str(idx).strip().lower(): idx for idx in PXRD_Data.index}
    Matched_shape = Shape_Map.get(Crystal_Shape.lower())

    if Matched_shape is None:
        print(f"Crystal Shape '{Crystal_Shape}' not found. Available values: {list(PXRD_Data.index)}")
        return pd.DataFrame()

    # NE_Needle_1 → NE_Needle, NE_Small_Square_2 → NE_Small_Square
    base_type = Matched_shape.rsplit("_", 1)[0]

    Fixed_PXRD_Values = PXRD_Data.loc[[Matched_shape]].reset_index(drop=True)

    if base_type in SEM_Data.index:
        Fixed_SEM_Values = SEM_Data.loc[[base_type]].reset_index(drop=True)
    else:
        print(f"No SEM data for '{base_type}'. SEM features will be NaN.")
        Fixed_SEM_Values = pd.DataFrame([[np.nan] * len(SEM_Data.columns)], columns=SEM_Data.columns)

    Prediction_Data_Conditions = pd.DataFrame({
        'Blending_Time': [Blending_Time],
        'LPM': [LPM]
    })

    All_Inputs = pd.concat([Prediction_Data_Conditions, Fixed_PXRD_Values, Fixed_SEM_Values], axis=1)

    # Filter to only the features the trained model knows about
    model_features = list(RF_NGI_Model.feature_names_in_)
    Prediction_Inputs = All_Inputs[model_features]

    # ============================
    # 2. Predict NGI Value
    # ============================
    Prediction_Outputs = RF_NGI_Model.predict(Prediction_Inputs)[0]
    Prediction_Results = pd.DataFrame([Prediction_Outputs], columns=Target_cols)

    return Prediction_Results
