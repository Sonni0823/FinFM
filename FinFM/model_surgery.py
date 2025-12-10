# c:\FinFMPrj\FinFM\model_surgery.py

import torch
import torch.nn as nn
import sys
import timesfm

# [1] 새로운 입력 블록 정의 (Custom Layer)
class MultivariateInputBlock(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, model_dim)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(model_dim, model_dim)
        # 차원이 다르므로 투영을 위한 Residual Layer
        self.residual_layer = nn.Linear(input_dim, model_dim)

    def forward(self, x):
        res = self.residual_layer(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x + res

# [2] 모델 수술 집도 함수
def perform_model_surgery(tfm_wrapper, num_features=4):
    print(">>> [Core Task 2] Performing Model Surgery...")
    
    if not hasattr(tfm_wrapper, 'model'):
        raise AttributeError("Wrapper does not have 'model' attribute.")
    
    internal_model = tfm_wrapper.model
    
    # ---------------------------------------------------------
    # Step 1: 백본 동결 (Backbone Freezing)
    # ---------------------------------------------------------
    print("  - [Step 1] Freezing Transformer Backbone (stacked_xf)...")
    if hasattr(internal_model, 'stacked_xf'):
        for param in internal_model.stacked_xf.parameters():
            param.requires_grad = False
    else:
        raise AttributeError("Cannot find 'stacked_xf' in model.")

    # ---------------------------------------------------------
    # Step 2: 입력 블록 교체 (Input Block Replacement)
    # ---------------------------------------------------------
    print("  - [Step 2] Replacing Input Tokenizer...")
    
    model_dim = 1280 
    patch_len = 32
    
    # [Fix] 마스크 차원(+1) 추가
    # TimesFM의 forward 함수가 내부적으로 input과 mask를 concat하므로
    # 실제 입력 차원은 (patch_len * num_features) + 1 이 됩니다.
    new_input_dim = (patch_len * num_features) + 1 
    
    new_tokenizer = MultivariateInputBlock(input_dim=new_input_dim, model_dim=model_dim)
    internal_model.tokenizer = new_tokenizer
    
    # ---------------------------------------------------------
    # [핵심 수정] Step 3: 출력 블록 교체 (Output Block Replacement)
    # ---------------------------------------------------------
    print("  - [Step 3] Replacing Output Projection (1280 -> 128)...")
    
    # 기존: ResidualBlock (1280 -> 1280) -> 에러 원인
    # 변경: Linear (1280 -> 128) -> 정답(Targets) 차원과 일치시킴
    horizon_len = 128
    
    # 단순한 Linear Layer로 교체하여 차원을 축소합니다.
    # 이것이 우리의 새로운 "Prediction Head"가 됩니다.
    internal_model.output_projection_point = nn.Linear(model_dim, horizon_len)
    
    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    trainable_params = sum(p.numel() for p in internal_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in internal_model.parameters())
    
    print(f"\n>>> [Surgery Complete] Model Status:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    print(f"  - New Input Dim: {new_input_dim}")
    print(f"  - New Output Dim: {horizon_len}")
    
    return tfm_wrapper

if __name__ == "__main__":
    # Test Surgery
    from timesfm import TimesFM_2p5_200M_torch
    print("Loading original model...")
    tfm = TimesFM_2p5_200M_torch(checkpoint="google/timesfm-1.0-200m")
    config = timesfm.ForecastConfig(per_core_batch_size=1, max_horizon=128, max_context=512)
    tfm.compile(config)
    tfm_modified = perform_model_surgery(tfm, num_features=4)
    print("\n>>> Verifying New Output Layer:")
    print(tfm_modified.model.output_projection_point)