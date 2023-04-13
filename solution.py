import pandas as pd
import numpy as np
import scipy.stats as stats

chat_id = 485082255 # Ваш chat ID, не меняйте название переменной

def solution(x_success: int, 
             x_cnt: int, 
             y_success: int, 
             y_cnt: int) -> bool:
    alpha = 0.03
    control_cr = x_success / x_cnt
    test_cr = y_success / y_cnt
    pooled_cr = (x_success + y_success) / (x_cnt + y_cnt)
    pooled_se = (pooled_cr * (1 - pooled_cr) * (1 / x_cnt + 1 / y_cnt)) ** 0.5
    z_score = (test_cr - control_cr) / pooled_se
    p_value = stats.norm.sf(z_score)
    
    return p_value < alpha   
