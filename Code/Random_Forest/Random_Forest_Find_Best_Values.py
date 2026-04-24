import pandas as pd
import numpy as np
import plotly.graph_objects as go

import joblib
from pathlib import Path

Workspace_root = Path(__file__).resolve().parents[2]

#==========================================================
# 1. Load Saved Model
#==========================================================

RF_NGI_Model = joblib.load(Workspace_root / "Model" / "NGI_Random_Forest_Model.pkl")
Target_cols = [
        "IP (mg)","PRE (mg)","Stage 1 (mg)","Stage 2 (mg)","Stage 3 (mg)","Stage 4 (mg)",
        "Stage 5 (mg)","Stage 6 (mg)","Stage 7 (mg)","Stage 8 (mg)",
        "Remaining (mg)","Mouth (mg)","PTP/Vial (mg)","ED (mg)","FPD (mg)","FPF (%)","RD (mg)","SUM_ALL (mg)",
    ]

FPD_Index = Target_cols.index('FPD (mg)')
FPF_Index = Target_cols.index('FPF (%)')

PXRD_Data = pd.read_csv(Workspace_root/'Data'/'Master'/'Selected_PXRD_Features.csv', index_col=0)


def Random_Forest_Find_Best_Values(crystal_shape, blending_time=None, lpm=None):
    Crystal_Shape = crystal_shape.strip()
    Shape_Map = {str(idx).strip().lower(): idx for idx in PXRD_Data.index}
    Matched_shape = Shape_Map.get(Crystal_Shape.lower())
    if Matched_shape is None:
        print(f"Crystal Shape '{Crystal_Shape}' not found. Available values: {list(PXRD_Data.index)}")
        return

    Fixed_PXRD_Values = PXRD_Data.loc[[Matched_shape]]

    Line_Space_LPM = 2
    Line_Space_Blend = 3
    LPM_Range = [40, 60]
    Blending_Time_Range = [30, 60, 90]
    if lpm is not None and str(lpm).strip() != "":
        v = float(str(lpm).strip().replace(",", "."))
        if v not in LPM_Range:
            LPM_Range = sorted(set(LPM_Range + [v]))
    if blending_time is not None and str(blending_time).strip() != "":
        v = float(str(blending_time).strip().replace(",", "."))
        if v not in Blending_Time_Range:
            Blending_Time_Range = sorted(set(Blending_Time_Range + [v]))

    LPM_Grid, Blend_Grid = np.meshgrid(LPM_Range, Blending_Time_Range)
    LPM_Flat = LPM_Grid.ravel()
    Blend_Flat = Blend_Grid.ravel()

    Simulation_Data = pd.DataFrame({
        'Blending_Time': Blend_Flat,
        'LPM': LPM_Flat
    })

    for Angle, Value in Fixed_PXRD_Values.items():
        Simulation_Data[Angle] = Value.iloc[0]

    Virtual_Experiment_Trial = Line_Space_Blend * Line_Space_LPM
    print("=====================================================")
    print(f"Prediction of {Virtual_Experiment_Trial} Virtual Experiments")
    print("=====================================================")

    Predicted_Outputs = RF_NGI_Model.predict(Simulation_Data)
    FPF_Predicted_Flat = Predicted_Outputs[:, FPF_Index]
    FPF_Grid = FPF_Predicted_Flat.reshape(LPM_Grid.shape)

    Best_Idx = np.argmax(FPF_Predicted_Flat)
    Best_LPM = LPM_Flat[Best_Idx]
    Best_Blend = Blend_Flat[Best_Idx]
    Max_FPF = FPF_Predicted_Flat[Best_Idx]

    print("=====================================================")
    print(f"Highest FPF(%) Condition of {Crystal_Shape}")
    print(f"Expected Value (%): {Max_FPF:.2f} %")
    print(f"Best LPM: {Best_LPM:.1f} L/min")
    print(f"Best Blending Time: {Best_Blend:.1f} min")
    print("=====================================================")

    print("=====================================================")
    print('Create & Save 3D Contour Heatmap Graph Using Plotly')
    print("=====================================================")
    Fig = go.Figure()

    Figure_Title = """Design Space of NGI Formulation\nCrystal Type: {Crystal_Shape})\nUsed Algorithm: Random Forest Regression"""

    Fig.add_trace(go.Surface(
        x=LPM_Grid,
        y=Blend_Grid,
        z=FPF_Grid,
        colorscale = 'Viridis',
        colorbar = dict(title='Predicted FPF(%)', len = 0.5),
        opacity = 0.9,
        name = Figure_Title
    ))

    Fig.add_trace(go.Scatter3d(
        x = [Best_LPM],
        y = [Best_Blend],
        z = [Max_FPF],
        mode = 'markers+text',
        marker = dict(
            size = 8,
            color = 'red',
            symbol = 'diamond',
            line = dict(width=2, color='darkred')
        ),
        text = [f'Max FPF: ({Max_FPF:.1f}%), LPM: ({Best_LPM:.2f}) L/min, Blending Time: ({Best_Blend:.2f}) min'],
        textposition = "top center",
        name = 'Optimal Point'
    ))

    Fig.update_layout(
        title = dict(
            text = f'''3D Design Space NGI Optimization
            Crystal Type = {Crystal_Shape}
            Used Algorithm: Random Forest Regression''',
            font = dict(size=20),
            x = 0.5
        ),
        scene=dict(
            xaxis_title='Airflow Rate (LPM)',
            yaxis_title='Blending Time (min)',
            zaxis_title='Predicted FPF (%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    Fig.show()
    HTML_Filename = 'FPF_Optimization_RF.html'
    Fig.write_html(Workspace_root / "Graph" / HTML_Filename)


if __name__ == "__main__":
    print("Input Crystal Shape (Ex. NE_Rod_1):")
    Crystal_Shape = input('').strip()
    Shape_Map = {str(idx).strip().lower(): idx for idx in PXRD_Data.index}
    while True:
        Matched_shape = Shape_Map.get(Crystal_Shape.lower())
        if Matched_shape is not None:
            break
        print(f"Crystal Shape '{Crystal_Shape}' not found. Available values: {list(PXRD_Data.index)}")
        print("Retry Crystal Shape:")
        Crystal_Shape = input('').strip()
    Random_Forest_Find_Best_Values(Crystal_Shape)
