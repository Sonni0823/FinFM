# c:\FinFMPrj\FinFM\finetune_dataset.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, 
                 csv_path="training_dataset_v1.csv", 
                 mode="train", 
                 context_len=512, 
                 horizon_len=128, 
                 patch_len=32,
                 train_ratio=0.8,
                 val_ratio=0.1):
        """
        [Core Task 1-2 & 1-3]
        - 1-2: Covariates Z-Score Normalization (Fit on Train)
        - 1-3: Multivariate Patch Generation
        """
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.patch_len = patch_len
        self.total_window = context_len + horizon_len
        
        # 1. 데이터 로드
        df = pd.read_csv(csv_path)
        
        # 컬럼 분리
        # Target: SP500_Close (첫번째 컬럼)
        # Covariates: 나머지 컬럼 (VIX, US_RATE, USD_KRW)
        target_col = 'SP500_Close'
        cov_cols = [c for c in df.columns if c != 'Date' and c != target_col]
        
        # 2. 데이터 분할 (Split)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        if mode == "train":
            self.data_df = df.iloc[:train_end].copy()
        elif mode == "val":
            self.data_df = df.iloc[train_end - self.total_window : val_end].copy() # 윈도우 연결성 보장
        elif mode == "test":
            self.data_df = df.iloc[val_end - self.total_window:].copy()
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")
            
        # 3. [Core Task 1-2] 공변량 Z-Score 정규화 (Dual Normalization Part 2)
        # 주의: 통계(mean, std)는 반드시 전체 데이터셋의 'Train' 구간에서만 계산해야 함
        train_df = df.iloc[:train_end]
        self.cov_mean = train_df[cov_cols].mean().values
        self.cov_std = train_df[cov_cols].std().values
        
        # 정규화 적용
        # (Target인 SP500은 RevIN을 위해 정규화하지 않고 Raw 값 유지)
        self.targets = self.data_df[target_col].values.astype(np.float32)
        self.covariates = self.data_df[cov_cols].values.astype(np.float32)
        
        self.covariates = (self.covariates - self.cov_mean) / (self.cov_std + 1e-6)
        
        print(f"[{mode.upper()}] Dataset Created. Shape: {self.targets.shape}")
        if mode == "train":
            print(f"  >>> Covariate Stats (Mean): {self.cov_mean}")
            print(f"  >>> Covariate Stats (Std):  {self.cov_std}")

    def __len__(self):
        return len(self.targets) - self.total_window + 1

    def __getitem__(self, idx):
        """
        [Core Task 1-3] Multivariate Patch Generation
        Returns:
            input_patches: (Num_Patches, Patch_Len, 1 + C)
            future_values: (Horizon_Len) - For Label
        """
        # 윈도우 슬라이싱
        window_start = idx
        context_end = idx + self.context_len
        horizon_end = context_end + self.horizon_len
        
        # 입력 데이터 (Context)
        past_target = self.targets[window_start : context_end]       # (512,)
        past_cov = self.covariates[window_start : context_end]       # (512, C)
        
        # 정답 데이터 (Horizon)
        future_target = self.targets[context_end : horizon_end]      # (128,)
        
        # 차원 결합 (Concatenation) -> (512, 1 + C)
        # Target을 (512, 1)로 변환 후 병합
        past_target_unsqueezed = past_target[:, np.newaxis]
        past_combined = np.concatenate([past_target_unsqueezed, past_cov], axis=1) # (512, 4)
        
        # 패치 생성 (Reshape)
        # Sequence Length(512)를 Patch Length(32)로 나눔 -> 16개 패치
        # Shape: (Num_Patches, Patch_Len, Features)
        num_patches = self.context_len // self.patch_len
        input_patches = past_combined.reshape(num_patches, self.patch_len, -1)
        
        # Tensor 변환
        return {
            "input_patches": torch.from_numpy(input_patches), # (16, 32, 4)
            "future_target": torch.from_numpy(future_target)  # (128,)
        }

if __name__ == "__main__":
    # Test Code
    print(">>> Testing FinancialTimeSeriesDataset...")
    
    # Train Dataset 생성
    ds_train = FinancialTimeSeriesDataset(mode="train")
    
    # 샘플 하나 가져오기
    sample = ds_train[0]
    input_patches = sample['input_patches']
    future_target = sample['future_target']
    
    print("\n>>> [Sample Verification]")
    print(f"Input Patches Shape: {input_patches.shape}") 
    print(f"  - Expected: (Num_Patches=16, Patch_Len=32, Features=4)")
    print(f"Future Target Shape: {future_target.shape}")
    print(f"  - Expected: (Horizon_Len=128,)")
    
    # 배치 테스트
    dl = DataLoader(ds_train, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print(f"\n>>> [Batch Verification]")
    print(f"Batch Input Shape: {batch['input_patches'].shape}")
    print(f"  - Expected: (8, 16, 32, 4)")