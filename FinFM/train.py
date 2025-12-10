# c:\FinFMPrj\FinFM\train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time

# 커스텀 모듈 임포트
import timesfm
import finetune_dataset
import model_surgery
from timesfm import TimesFM_2p5_200M_torch

# [설정] 하이퍼파라미터
CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 20,
    "weight_decay": 1e-2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 5
}

def train_model():
    print("="*60)
    print(">>> [Core Task 3] Starting TimesFM Finetuning Pipeline")
    print(f">>> Device: {CONFIG['device']}")
    print("="*60)

    # 1. 데이터셋 및 데이터로더 준비
    print("\n[Step 1] Loading Datasets...")
    train_ds = finetune_dataset.FinancialTimeSeriesDataset(mode="train")
    val_ds = finetune_dataset.FinancialTimeSeriesDataset(mode="val")
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    print(f"  - Train Batches: {len(train_loader)}")
    print(f"  - Val Batches: {len(val_loader)}")

    # 2. 모델 로드 및 수술
    print("\n[Step 2] Loading & Modifying Model...")
    tfm = TimesFM_2p5_200M_torch(checkpoint="google/timesfm-1.0-200m")
    
    # 컴파일
    tfm_config = timesfm.ForecastConfig(per_core_batch_size=1, max_horizon=128, max_context=512)
    tfm.compile(tfm_config)
    
    # 수술 집도
    tfm = model_surgery.perform_model_surgery(tfm, num_features=4)
    
    # 모델 추출 및 설정
    model = tfm.model
    model.to(CONFIG['device'])
    model.train()

    # 3. 옵티마이저 및 손실함수
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.MSELoss()

    # 4. 학습 루프
    print("\n[Step 3] Starting Training Loop...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        # --- Training ---
        model.train()
        train_loss_accum = 0.0
        
        for batch in train_loader:
            # Data Type Fixing: float32
            inputs = batch['input_patches'].to(CONFIG['device']).float()
            targets = batch['future_target'].to(CONFIG['device']).float()
            
            # [RevIN Logic]
            target_context = inputs[..., 0].view(inputs.size(0), -1) 
            mean = target_context.mean(dim=1, keepdim=True)
            std = target_context.std(dim=1, keepdim=True) + 1e-6
            
            inputs[..., 0] = (inputs[..., 0] - mean.unsqueeze(-1)) / std.unsqueeze(-1)
            targets_norm = (targets - mean) / std
            
            # Flatten Inputs
            B, N, P, F = inputs.shape
            inputs_flat = inputs.view(B, N, -1)
            
            # Mask Creation
            padding_masks = torch.zeros((B, N), dtype=torch.float32).to(CONFIG['device'])
            padding_masks = padding_masks.unsqueeze(-1) 
            
            # Forward Pass
            optimizer.zero_grad()
            
            outputs = model(inputs_flat, padding_masks)
            
            # Robust Unwrapping
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # [Fix] Projection Layer 통과 (1280 -> 128)
            # Transformer 출력(1280)을 예측값(128)으로 변환
            last_patch_hidden = outputs[:, -1, :] # (B, 1280)
            last_patch_pred = model.output_projection_point(last_patch_hidden) # (B, 128)

            loss = criterion(last_patch_pred, targets_norm)
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_patches'].to(CONFIG['device']).float()
                targets = batch['future_target'].to(CONFIG['device']).float()
                
                # Validation RevIN
                target_context = inputs[..., 0].view(inputs.size(0), -1)
                mean = target_context.mean(dim=1, keepdim=True)
                std = target_context.std(dim=1, keepdim=True) + 1e-6
                
                inputs[..., 0] = (inputs[..., 0] - mean.unsqueeze(-1)) / std.unsqueeze(-1)
                targets_norm = (targets - mean) / std
                
                inputs_flat = inputs.view(inputs.size(0), 16, -1)
                
                # Mask Creation
                B, N, _ = inputs_flat.shape
                padding_masks = torch.zeros((B, N), dtype=torch.float32).to(CONFIG['device'])
                padding_masks = padding_masks.unsqueeze(-1)
                
                outputs = model(inputs_flat, padding_masks)
                
                # Robust Unwrapping
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # [Fix] Projection Layer 통과
                last_patch_hidden = outputs[:, -1, :]
                last_patch_pred = model.output_projection_point(last_patch_hidden)
                
                loss = criterion(last_patch_pred, targets_norm)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} "
              f"| Time: {epoch_time:.1f}s")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print("  >>> Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"  >>> Early stopping triggered after {epoch+1} epochs.")
                break

    print("\n>>> Finetuning Complete.")
    print(f">>> Best Validation Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    train_model()