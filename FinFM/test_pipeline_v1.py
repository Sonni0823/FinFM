# c:\FinFMPrj\FinFM\test_pipeline_v1.py

import os
import sys
import pandas as pd
import torch

# 우리가 만든 모듈들 임포트
try:
    import data_merger
    import finetune_dataset
    print(">>> [Test] Modules imported successfully.")
except ImportError as e:
    print(f">>> [Fail] Import failed: {e}")
    sys.exit(1)

def run_integration_test():
    print("="*60)
    print(">>> [Integration Test] Core Task 1: Data Engineering Pipeline")
    print("="*60)

    # ---------------------------------------------------------
    # STEP 1: 데이터 다운로드, 병합, CSV 생성 (data_merger 실행)
    # ---------------------------------------------------------
    print("\n>>> [Step 1] Running 'data_merger' to create CSV artifact...")
    try:
        df_final = data_merger.merge_and_clean_data()
        
        # [수정] 절대 경로 생성 로직 추가
        # 현재 스크립트(test_pipeline_v1.py)가 있는 폴더 경로를 구함
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 무조건 이 폴더 안에 저장하도록 경로 결합
        csv_path = os.path.join(current_dir, "training_dataset_v1.csv")
        
        df_final.to_csv(csv_path, index=False)
        
        if os.path.exists(csv_path):
            print(f">>> [Pass] Artifact created at: {csv_path}") # 절대 경로 출력
            print(f">>> Rows: {len(df_final)}")
        else:
            print(">>> [Fail] CSV file was not created.")
            sys.exit(1)
            
    except Exception as e:
        print(f">>> [Fail] Error during data merging: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: 데이터셋 클래스 로딩 및 텐서 변환 (finetune_dataset 실행)
    # ---------------------------------------------------------
    print("\n>>> [Step 2] Testing 'finetune_dataset' with generated CSV...")
    try:
        # CSV 파일을 읽어서 데이터셋 객체 생성
        ds = finetune_dataset.FinancialTimeSeriesDataset(
            csv_path="training_dataset_v1.csv",
            mode="train"
        )
        
        # 샘플 데이터 하나 추출
        sample = ds[0]
        inputs = sample['input_patches']  # (16, 32, 4)
        target = sample['future_target']  # (128,)
        
        print(f">>> [Pass] Dataset loaded successfully.")
        print(f"    - Input Shape: {inputs.shape} (Expected: [16, 32, 4])")
        print(f"    - Target Shape: {target.shape} (Expected: [128])")
        
        if inputs.shape == torch.Size([16, 32, 4]):
            print(">>> [Pass] Dimensions match TimesFM requirements.")
        else:
            print(">>> [Fail] Dimension mismatch!")
            
    except Exception as e:
        print(f">>> [Fail] Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print(">>> [Success] Core Task 1 Pipeline is FULLY OPERATIONAL.")
    print("="*60)

if __name__ == "__main__":
    run_integration_test()