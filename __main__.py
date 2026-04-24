import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import runpy

def main():
    Workspace_root = Path(__file__).resolve().parent
    
    try:
        pxrd_data = pd.read_csv(Workspace_root / 'Data' / 'Master' / 'Selected_PXRD_Features.csv', index_col=0)
        valid_shapes = list(pxrd_data.index)
        valid_shapes_lower = {str(s).strip().lower(): str(s) for s in valid_shapes}
    except Exception:
        valid_shapes = []
        valid_shapes_lower = {}

    while True:
        print("=====================================================")
        print("NGI Prediction Model")
        print("=====================================================")
        print("Choose Option:")
        print("1. Model Training & File Export")
        print("2. Prediction")  
        print("3. Find Best Value")
        print("4. Analyze SEM Image")
        print("0. End")
        print("=====================================================")
        choice = input("Input Option Number: ")
        print("=====================================================")

        if choice == "1":
            print("=====================================================")
            print("Choose Prediction Model:")
            print("1. Random Forest")
            print("2. MLP")
            print("0. Return to Main Menu")
            print("=====================================================")
            model_choice = input("Input Model Number: ")
            if model_choice == "1":
                runpy.run_module("Code.Random_Forest.Random_Forest_Training", run_name="__main__")
            elif model_choice == "2":
                runpy.run_module("Code.MLP.MLP_Training", run_name="__main__")
            elif model_choice == "0":
                print("Return to Main Menu.")
                continue
            
            else:
                print("Unknown Option!")
            

        elif choice == "2":
            print("=====================================================")
            print("Choose Prediction Model:")
            print("1. Random Forest")
            print("2. MLP")
            print("0. Return to Main Menu")
            print("=====================================================")
            model_choice = input("Input Model Number: ")
            if model_choice == "0":
                print("Return to Main Menu.")
                continue
            elif model_choice not in ["1", "2"]:
                print("Unknown Option!")
                continue

            while True:
                Crystal_Shape = input("Insert Crystal Shape (Default. NE_Rod_1): ").strip()
                if Crystal_Shape == "":
                    Crystal_Shape = "NE_Rod_1"
                if valid_shapes_lower and Crystal_Shape.lower() not in valid_shapes_lower:
                    print(f"Crystal Shape '{Crystal_Shape}' not found. Available values: {valid_shapes}")
                    continue
                if valid_shapes_lower:
                    Crystal_Shape = valid_shapes_lower[Crystal_Shape.lower()]
                break
            Blending_Time_str = input("Insert Blending Time (Default: 30): ")
            Blending_Time = int(Blending_Time_str) if Blending_Time_str.strip() else 30
            LPM_str = input("Insert LPM (Default: 60): ")
            LPM = int(LPM_str) if LPM_str.strip() else 60
            
            if model_choice == "1":
                import Code.Random_Forest.Random_Forest_Prediction as rf_pred
                importlib.reload(rf_pred)
                _result = rf_pred.Prediction_Data_Prepare(Crystal_Shape, Blending_Time, LPM)
            elif model_choice == "2":
                import Code.MLP.MLP_Prediction as mlp_pred
                importlib.reload(mlp_pred)
                _result = mlp_pred.Prediction_Data_Prepare(Crystal_Shape, Blending_Time, LPM)
                
            print(f"""Prediction of FPF(%)
Crystal: {Crystal_Shape}
Blending Time: {Blending_Time}
LPM: {LPM}
Algorithm: {model_choice}
FPF(%): {_result['FPF (%)'].iloc[0]}""")

        elif choice == "3":
            print("=====================================================")
            print("Choose Prediction Model:")
            print("1. Random Forest")
            print("2. MLP")
            print("0. Return to Main Menu")
            print("=====================================================")
            model_choice = input("Input Model Number: ")
            if model_choice == "0":
                print("Return to Main Menu.")
                continue
            elif model_choice not in ["1", "2"]:
                print("Unknown Option!")
                continue

            while True:
                Crystal_Shape = input("Insert Crystal Shape (Ex. NE_Rod_1): ").strip()
                if Crystal_Shape == "":
                    Crystal_Shape = "NE_Rod_1"
                if valid_shapes_lower and Crystal_Shape.lower() not in valid_shapes_lower:
                    print(f"Crystal Shape '{Crystal_Shape}' not found. Available values: {valid_shapes}")
                    continue
                if valid_shapes_lower:
                    Crystal_Shape = valid_shapes_lower[Crystal_Shape.lower()]
                break
            Blending_Time_str = input("Insert Blending Time (Default: 30): ")
            Blending_Time = int(Blending_Time_str) if Blending_Time_str.strip() else 30
            LPM_str = input("Insert LPM (Default: 60): ")
            LPM = int(LPM_str) if LPM_str.strip() else 60
            
            if model_choice == "1":
                import Code.Random_Forest.Random_Forest_Find_Best_Values as rf_best
                importlib.reload(rf_best)
                rf_best.Random_Forest_Find_Best_Values(Crystal_Shape, Blending_Time, LPM)
            elif model_choice == "2":
                import Code.MLP.MLP_Find_Best_Values as mlp_best
                importlib.reload(mlp_best)
                mlp_best.MLP_find_best_values(Crystal_Shape, Blending_Time, LPM)

        elif choice == "4":
            runpy.run_module("Code.SEM.SEM_Crystal_Analyzer", run_name="__main__")
        
        elif choice == "0":
            print("Terminate program.")
            break

        else:
            print("Unknown Option!")

if __name__ == "__main__":
    main()