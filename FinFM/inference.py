# c:\FinFMPrj\FinFM\inference.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 커스텀 모듈 임포트
import timesfm
import finetune_dataset
import model_surgery
from timesfm import TimesFM_2p5_200M_torch

# [설정]
CONFIG = {
    "batch_size": 1,           # 추론은 하나씩 또는 배치로 가능
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "best_finetuned_model.pth",
    "horizon_len": 128
}

def load_trained_model():
    print(">>> Loading Trained Model...")
    
    # 1. 구조 생성 (학습 때와 동일해야 함)
    tfm = TimesFM_2p5_200M_torch(checkpoint="google/timesfm-1.0-200m")
    tfm_config = timesfm.ForecastConfig(per_core_batch_size=1, max_horizon=128, max_context=512)
    tfm.compile(tfm_config)
    
    # 2. 수술 (Surgery)
    tfm = model_surgery.perform_model_surgery(tfm, num_features=4)
    model = tfm.model
    
    # 3. 가중치 로드 (State Dict)
    if os.path.exists(CONFIG['checkpoint_path']):
        state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
        print(">>> Weights loaded successfully from:", CONFIG['checkpoint_path'])
    else:
        print(">>> [Warning] Checkpoint not found! Using random weights.")
    
    model.to(CONFIG['device'])
    model.eval()
    return model

def run_inference_and_plot():
    model = load_trained_model()
    
    # Test Dataset 로드 (학습에 안 쓴 데이터)
    test_ds = finetune_dataset.FinancialTimeSeriesDataset(mode="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f">>> Test Samples: {len(test_ds)}")
    
    # 시각화할 샘플 인덱스 (예: 데이터셋의 중간 쯤)
    sample_indices = [0, len(test_ds)//2, len(test_ds)-1]
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i not in sample_indices:
                continue
                
            print(f"  - Predicting Sample {i}...")
            
            inputs = batch['input_patches'].to(CONFIG['device']).float()
            targets = batch['future_target'].to(CONFIG['device']).float()
            
            # [RevIN Statistics Calculation]
            # 역정규화를 위해 저장해둠
            target_context = inputs[..., 0].view(inputs.size(0), -1)
            mean = target_context.mean(dim=1, keepdim=True)
            std = target_context.std(dim=1, keepdim=True) + 1e-6
            
            # Input Normalize
            inputs[..., 0] = (inputs[..., 0] - mean.unsqueeze(-1)) / std.unsqueeze(-1)
            
            # Flatten & Mask
            B, N, P, F = inputs.shape
            inputs_flat = inputs.view(B, N, -1)
            padding_masks = torch.zeros((B, N), dtype=torch.float32).to(CONFIG['device']).unsqueeze(-1)
            
            # Inference
            outputs = model(inputs_flat, padding_masks)
            
            # Robust Unwrapping
            if isinstance(outputs, (tuple, list)): outputs = outputs[0]
            if isinstance(outputs, (tuple, list)): outputs = outputs[0]
            
            last_patch_hidden = outputs[:, -1, :]
            pred_norm = model.output_projection_point(last_patch_hidden) # (1, 128)
            
            # [Denormalization] (정규화된 값 -> 실제 주가)
            # Pred * Std + Mean
            pred_real = pred_norm * std + mean
            target_real = targets # Target은 원래 Raw값이었음 (Dataset에서 정규화 안함)
            
            # CPU로 이동
            pred_real = pred_real.cpu().numpy().flatten()
            target_real = target_real.cpu().numpy().flatten()
            history_real = target_context.cpu().numpy().flatten() # 문맥 데이터
            
            results.append({
                "idx": i,
                "history": history_real,
                "truth": target_real,
                "pred": pred_real
            })

    # [Plotting]
    plot_results(results)

def plot_results(results):
    print(">>> Plotting Results...")
    num_samples = len(results)
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 5 * num_samples))
    
    if num_samples == 1: axes = [axes]
    
    for ax, res in zip(axes, results):
        idx = res['idx']
        history = res['history']
        truth = res['truth']
        pred = res['pred']
        
        # X축 생성
        # History: -512 ~ -1
        # Future: 0 ~ 127
        x_hist = np.arange(-len(history), 0)
        x_future = np.arange(0, len(truth))
        
        # Plot History (문맥)
        ax.plot(x_hist, history, label='History (Context)', color='gray', alpha=0.5)
        
        # Plot History의 마지막 점과 Future의 첫 점 연결 (시각적 연속성)
        connect_x = [x_hist[-1], x_future[0]]
        connect_y_truth = [history[-1], truth[0]]
        connect_y_pred = [history[-1], pred[0]]
        ax.plot(connect_x, connect_y_truth, color='green', linestyle='--')
        ax.plot(connect_x, connect_y_pred, color='red', linestyle='--')

        # Plot Ground Truth (정답)
        ax.plot(x_future, truth, label='Ground Truth', color='green', linewidth=2)
        
        # Plot Prediction (예측)
        ax.plot(x_future, pred, label='TimesFM Prediction', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f"S&P 500 Forecast (Test Sample #{idx}) - Horizon: 128 Days")
        ax.set_xlabel("Time Steps (Days)")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 주요 구간 줌인 (마지막 30일 + 미래)
        ax.set_xlim(-60, 130) 

    plt.tight_layout()
    plt.savefig("forecast_result.png")
    print(">>> Plot saved to: forecast_result.png")
    # plt.show() # 로컬 환경에서 창을 띄우려면 주석 해제

import os
if __name__ == "__main__":
    if not os.path.exists("best_finetuned_model.pth"):
        print(">>> [Wait] Training is not finished yet. Run this after training.")
    else:
        run_inference_and_plot()