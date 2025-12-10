# check_attributes.py
import timesfm.timesfm_2p5.timesfm_2p5_torch as tfm_module

print(">>> [Debug] Attributes in 'timesfm_2p5_torch':")
# 모듈 내의 모든 속성 중 대문자로 시작하는 것(클래스일 확률 높음)만 출력
for attr in dir(tfm_module):
    if attr[0].isupper():
        print(f" - {attr}")