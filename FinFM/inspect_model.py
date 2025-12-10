# c:\FinFMPrj\FinFM\inspect_model.py

import timesfm
import torch
import sys

def inspect_model_structure():
    print(">>> [Core Task 2-0] Inspecting Model Architecture...")
    
    # 1. 모델 로드 (체크포인트 없이 구조만 로드해도 됨)
    try:
        from timesfm import TimesFM_2p5_200M_torch
        tfm = TimesFM_2p5_200M_torch()
        # 구조 확인을 위해 최소 컴파일
        config = timesfm.ForecastConfig(
            per_core_batch_size=1,
            max_horizon=128,
            max_context=512
        )
        tfm.compile(config)
    except Exception as e:
        print(f">>> [Error] Model loading failed: {e}")
        sys.exit(1)

    print("\n>>> [Model Anatomy Report]")
    print(f"Model Class: {type(tfm)}")
    
    # 2. 내부 PyTorch 모듈 확인
    # TimesFM 2.5 클래스는 내부에 실제 torch.nn.Module을 감싸고 있을 수 있음.
    # 보통 .model, ._model, .patches_model 등의 이름으로 존재함.
    
    target_model = None
    
    # 예상되는 내부 모듈 이름 탐색
    potential_names = ['model', '_model', 'p_model', 'stacked_transformer']
    
    for name in potential_names:
        if hasattr(tfm, name):
            print(f"Found internal module attribute: '{name}'")
            target_model = getattr(tfm, name)
            break
            
    if target_model is None:
        print(">>> [Warning] Could not find internal torch module by name.")
        print(">>> Dumping dir(tfm) to help identify:")
        print([d for d in dir(tfm) if not d.startswith('__')])
        
        # 만약 tfm 자체가 nn.Module이라면
        if isinstance(tfm, torch.nn.Module):
            target_model = tfm
    
    if target_model:
        print(f"\n>>> Analyzing Layers in '{type(target_model).__name__}':")
        
        # 전체 레이어 요약 출력
        print("-" * 60)
        print(f"{'Layer Name':<40} | {'Type':<20}")
        print("-" * 60)
        
        for name, module in target_model.named_modules():
            # 너무 깊은 서브모듈은 제외하고 최상위 레벨 위주로 출력
            if name.count('.') <= 2: 
                print(f"{name:<40} | {module.__class__.__name__:<20}")
                
        print("-" * 60)
        
        print("\n>>> [Diagnostic Action Items]")
        print("1. Look for 'Input Block': names containing 'input', 'residual', 'embed'")
        print("2. Look for 'Output Block': names containing 'output', 'proj', 'head'")
        print("3. Look for 'Transformer': names containing 'blocks', 'layers', 'attn'")

if __name__ == "__main__":
    inspect_model_structure()