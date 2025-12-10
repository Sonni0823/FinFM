# c:\FinFMPrj\FinFM\data_merger.py

import pandas as pd
import numpy as np
import data_loader  # [중요] 이전 파일명을 data_loader.py로 변경해야 임포트 가능

def merge_and_clean_data():
    """
    [Core Task 1-1-2 & 1-1-3]
    정보 이월(Carry-over) 병합 및 결측치 처리를 수행합니다.
    """
    print(">>> [Core Task 1-1-2] Starting Data Merging & Cleaning...")

    # 1. 데이터 로드 (이전 단계 모듈 재사용)
    dfs = data_loader.load_and_standardize_data()
    
    # 2. 기준점(Master Index) 설정: S&P 500 거래일
    df_master = dfs['TARGET'].sort_values('Date').copy()
    
    # 3. 공통 시작일(Common Start Date) 찾기 (Truncation)
    # USD_KRW가 2003년부터 시작하므로, 그 이전 데이터는 NaN이 됩니다.
    # 학습 안정성을 위해 가장 늦게 시작하는 데이터 기준으로 앞부분을 잘라냅니다.
    latest_start_date = max([df['Date'].min() for df in dfs.values()])
    print(f">>> [Info] Truncating data before common start date: {latest_start_date}")
    
    df_master = df_master[df_master['Date'] >= latest_start_date]

    # 4. 반복적 병합 (Iterative Merge with merge_asof)
    covariates = ["VIX", "US_RATE", "USD_KRW"]
    
    for name in covariates:
        df_cov = dfs[name].sort_values('Date')
        
        # [핵심 로직] merge_asof (Backward)
        # S&P 500 거래일(t) 기준으로, t와 같거나 가장 가까운 과거(t-k)의 지표를 가져옴.
        # 즉, 주말/휴일 동안 발표된 지표가 월요일/개장일 데이터로 매핑됨.
        df_master = pd.merge_asof(
            df_master,
            df_cov,
            on='Date',
            direction='backward',
            suffixes=('', f'_{name}') # 충돌 방지 (사실 이름이 달라서 필요 없으나 안전장치)
        )
        print(f"  - Merged {name} using backward carry-over.")

    # 5. 결측치 처리 (Handling Missing Values)
    # merge_asof 이후에도 중간중간(Intraday missing) 결측이 있을 수 있음.
    # 예: 채권시장만 휴장이라 US_RATE가 없는 날.
    
    # 전략: Forward Fill (직전 값 유지)
    # 경고: Backward Fill은 절대 금지 (Look-ahead Bias)
    initial_rows = len(df_master)
    df_master = df_master.ffill()
    
    # ffill로도 채워지지 않는 초반 행(Start-point gaps) 제거
    df_master = df_master.dropna()
    final_rows = len(df_master)
    
    if initial_rows != final_rows:
        print(f">>> [Info] Dropped {initial_rows - final_rows} rows due to initial NaNs.")

    # 6. 최종 검증
    print(">>> Final Data Integrity Check:")
    print(df_master.info())
    print(df_master.head())
    print(df_master.tail())

    return df_master

if __name__ == "__main__":
    df_final = merge_and_clean_data()
    
    # 결과 저장 (Artifact)
    df_final.to_csv("training_dataset_v1.csv", index=False)
    print(">>> Artifact saved: training_dataset_v1.csv")