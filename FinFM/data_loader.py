# c:\FinFMPrj\FinFM\data_loader.py

import yfinance as yf
import pandas as pd
import pytz

def load_and_standardize_data(start_date="2000-01-01", end_date=None):
    """
    S&P 500 및 거시 경제 지표를 로드하고 시간대를 미국 동부 시간(US/Eastern)으로 통일합니다.
    """
    print(">>> [Core Task 1-1-1] Starting Data Loading & Timezone Standardization...")

    # 1. 티커 정의
    # Target: S&P 500 (^GSPC)
    # Covariates: 
    #   - VIX 지수 (^VIX)
    #   - 미국 10년물 국채 금리 (^TNX) - 금리 대용
    #   - 원/달러 환율 (KRW=X)
    tickers = {
        "TARGET": "^GSPC",
        "VIX": "^VIX",
        "US_RATE": "^TNX",
        "USD_KRW": "KRW=X"
    }

    data_frames = {}

    for name, ticker in tickers.items():
        print(f"  - Downloading {name} ({ticker})...")
        
        # yfinance로 데이터 다운로드 (일일 데이터)
        # auto_adjust=True: 배당/분할 조정된 종가 사용
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # 멀티 인덱스 컬럼 문제 해결 (yfinance 최신 버전 호환)
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        # 2. 인덱스 초기화 및 Datetime 변환
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        # 3. 타임존 표준화 (Timezone Standardization)
        # 목표: 모든 시간을 'US/Eastern' (EST/EDT) 기준 16:00:00(장 마감)으로 통일
        
        # 3-1. Timezone 정보가 있는 경우 제거 (tz_localize를 위해)
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
        
        # 3-2. 'US/Eastern'으로 설정하고 시간(Hour) 정규화
        # 금융 데이터는 보통 장 마감 기준이므로 날짜만 중요하지만, 
        # TimesFM 처리를 위해 명확한 타임스탬프를 부여합니다.
        df['Date'] = df['Date'].dt.normalize() # 시간을 00:00:00으로 초기화
        
        # 필요한 컬럼만 선택 및 이름 변경
        if name == "TARGET":
            df = df[['Date', 'Close']].rename(columns={'Close': 'SP500_Close'})
        else:
            df = df[['Date', 'Close']].rename(columns={'Close': name})
            
        data_frames[name] = df

    print(">>> Data loading complete.")
    return data_frames

def validate_dates(data_frames):
    """
    로드된 데이터의 날짜 범위 및 타입을 검증합니다.
    """
    print(">>> Validating data types and ranges...")
    for name, df in data_frames.items():
        dtype = df['Date'].dtype
        start = df['Date'].min().strftime('%Y-%m-%d')
        end = df['Date'].max().strftime('%Y-%m-%d')
        count = len(df)
        print(f"  [{name}] Type: {dtype}, Range: {start} ~ {end}, Rows: {count}")
        
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            print(f"  [Error] {name} 'Date' column is not datetime object!")
            return False
            
    print(">>> Validation Passed.")
    return True

if __name__ == "__main__":
    # 테스트 실행
    dfs = load_and_standardize_data()
    validate_dates(dfs)
    
    # 결과 미리보기
    print("\n[Preview: S&P 500]")
    print(dfs['TARGET'].head())
    print("\n[Preview: VIX]")
    print(dfs['VIX'].head())