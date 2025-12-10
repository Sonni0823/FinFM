import timesfm
import numpy as np
import sys
import torch
import inspect

def load_timesfm_model():
    print(">>> [Core Task 1-1] Loading TimesFM model parameters...")
    
    # 1. 클래스 임포트
    try:
        from timesfm import TimesFM_2p5_200M_torch
        print(">>> [Info] Class 'TimesFM_2p5_200M_torch' imported successfully.")
    except ImportError:
        try:
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
            print(">>> [Info] Class imported from submodule.")
        except ImportError:
            print(">>> [Error] Could not import 'TimesFM_2p5_200M_torch'. Check installation.")
            sys.exit(1)

    # 2. 모델 인스턴스화
    try:
        tfm = TimesFM_2p5_200M_torch(checkpoint="google/timesfm-1.0-200m")
    except Exception as e:
        print(f">>> [Warning] Instantiation failed: {e}")
        tfm = TimesFM_2p5_200M_torch()

    # 3. ForecastConfig 설정 및 컴파일
    print(">>> [Info] Compiling model... (This may take a moment)")
    
    try:
        config_args = {
            "per_core_batch_size": 1,
            "max_horizon": 128,
            "max_context": 512,
        }
        
        # 유효성 검사 (안전 장치)
        sig = inspect.signature(timesfm.ForecastConfig)
        valid_args = {k: v for k, v in config_args.items() if k in sig.parameters}
        
        if len(valid_args) < len(config_args):
            print(f">>> [Debug] Removed invalid args. Using only: {valid_args.keys()}")

        config = timesfm.ForecastConfig(**valid_args)
        
        tfm.compile(config) 
        print(">>> [Info] Model compiled successfully.")
        
    except Exception as e:
        print(f">>> [Error] Compilation failed: {e}")
        print(">>> [Debug Info] Required arguments:", inspect.signature(timesfm.ForecastConfig))
        sys.exit(1)

    print(">>> Model object ready.")
    return tfm

def sanity_check(tfm):
    print(">>> Performing sanity check inference...")
    
    # 더미 데이터 (Batch=1, Length=512)
    dummy_input = np.random.rand(1, 512).astype(np.float32)
    
    try:
        # 추론 실행
        forecast_result = tfm.forecast(
            inputs=dummy_input,
            horizon=128
        )
        
        # 결과 처리
        if isinstance(forecast_result, tuple):
             # (point_forecast, quantile_forecast) 형태일 수 있음
             print(f">>> Forecast Shape: {forecast_result[0].shape}")
        else:
             print(f">>> Forecast Shape: {forecast_result.shape}")
             
        print(">>> Sanity Check Passed.")
        
    except Exception as e:
        print(f">>> [Error] Inference failed during sanity check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model = load_timesfm_model()
    sanity_check(model)