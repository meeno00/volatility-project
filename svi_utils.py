import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import os
from typing import Dict, List, Tuple, Callable, Any, Optional
from pricing import black_scholes_price


# -------------------------------
# 단계 0: 전처리
# -------------------------------
def preprocess_option_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 필수 값 존재 확인
    df = df.dropna(subset=['mark_iv', 'underlying_price', 'expiration', 'mark_price'])

    # 마크 가격 필터
    df = df[df['mark_price'] >= 0.001]

    # ITM 제거
    df = df[~(((df['type'] == 'call') & (df['strike_price'] < df['underlying_price'])) |
              ((df['type'] == 'put') & (df['strike_price'] > df['underlying_price'])))]

    # 시간 계산
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['expiration'] = pd.to_datetime(df['expiration'], utc=True)
    df['T'] = (df['expiration'] - df['timestamp']).dt.total_seconds() / (365 * 24 * 3600)

    # log-moneyness 및 total variance 계산
    df['F'] = df['underlying_price']
    df['k'] = np.log(df['strike_price'] / df['F'])
    df['mark_iv_decimal'] = df['mark_iv'] / 100
    df['total_variance'] = df['mark_iv_decimal'] ** 2 * df['T']

    # 유효성 필터링
    df = df[df['total_variance'] > 0]
    df = df[df['T'] > 0]

    # # 만기 15일 이하인 경우, strike_price가 2000 또는 5000 단위 인 것만 사용
    # short_term_mask = df['T'] <= (15 / 365)
    # round_strike_mask = (df['strike_price'] % 2000 == 0) | (df['strike_price'] % 5000 == 0)
    # df = df[~short_term_mask | round_strike_mask]

    # df['mark_price_usd'] = df['mark_price'] * df['index_price']
    df['mark_price_usd'] = df['mark_price'] * df['F']

    return df[['expiration', 'T', 'strike_price', 'F', 'k', 'mark_price_usd', 'mark_iv_decimal', 'total_variance', 'type']]


# -------------------------------
# 단계 1: Square-root SVI 초기화
# -------------------------------
def square_root_svi_w(k: np.ndarray, theta: np.ndarray, rho: float, eta: float) -> np.ndarray:
    phi = eta / (np.sqrt(theta) + 1e-8)  # 수치 안정성 보정
    term1 = 1 + rho * phi * k
    term2 = np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
    return theta * (term1 + term2)


def fit_square_root_svi(df: pd.DataFrame) -> Dict[str, Any]:
    grouped = df.groupby('expiration')
    theta_by_expiry = {}
    slice_data = {}

    for expiry, group in grouped:
        group = group.copy()
        group['abs_k'] = np.abs(group['k'])
        atm_row = group.loc[group['abs_k'].idxmin()]
        theta_by_expiry[expiry] = atm_row['total_variance']
        # slice_data[expiry] = group[['k', 'total_variance']]
        slice_data[expiry] = group[['k', 'total_variance', 'T']]

    all_k, all_w, all_theta, all_T = [], [], [], []
    for expiry, data in slice_data.items():
        theta = theta_by_expiry[expiry]
        all_k.extend(data['k'].tolist())
        all_w.extend(data['total_variance'].tolist())
        all_T.extend(data['T'].tolist())
        all_theta.extend([theta] * len(data))

    all_k, all_w, all_theta, all_T = map(np.array, [all_k, all_w, all_theta, all_T])

    # def svi_loss(params: Tuple[float, float]) -> float:
    #     rho, eta = params
    #     if not (-0.999 < rho < 0.999 and eta > 0): return np.inf
    #     w_model = square_root_svi_w(all_k, all_theta, rho, eta)
    #     if not np.all(np.isfinite(w_model)): return np.inf
    #     return np.mean((w_model - all_w) ** 2)
    
    def svi_loss(params: Tuple[float, float]) -> float:
        rho, eta = params
        if not (-0.999 < rho < 0.999 and eta > 0): return np.inf
        w_model = square_root_svi_w(all_k, all_theta, rho, eta)
        iv_model = np.sqrt(w_model / all_T)
        iv_market = np.sqrt(all_w / all_T)
        if not np.all(np.isfinite(w_model)): return np.inf
        return np.mean((iv_model - iv_market) ** 2)

    result = minimize(svi_loss, [0.0, 0.5], bounds=[(-0.999, 0.999), (1e-4, 5.0)])
    fitted_rho, fitted_eta = result.x

    return {
        'rho': fitted_rho,
        'eta': fitted_eta,
        'theta_by_expiry': theta_by_expiry,
        'svi_func': lambda k, T: square_root_svi_w(k, T, fitted_rho, fitted_eta)
    }


# -------------------------------
# 단계 2: QR SVI 피팅
# -------------------------------
def raw_svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


# def fit_raw_svi_slice(k: np.ndarray, w: np.ndarray, initial_params: List[float]) -> np.ndarray:
#     bounds = Bounds([-1.0, 0.0001, -0.999, -5.0, 0.0001], [1.0, 10.0, 0.999, 5.0, 5.0])

#     def loss(params: List[float]) -> float:
#         a, b, rho, m, sigma = params
#         model = raw_svi_total_variance(k, a, b, rho, m, sigma)
#         return np.mean((model - w) ** 2)
#     return minimize(loss, initial_params, bounds=bounds, method='L-BFGS-B').x

# === 핵심 수정 함수 ===
def fit_raw_svi_slice(
    # 기존 인수
    k: np.ndarray,
    w_market: np.ndarray,
    initial_params: List[float],
    # 새로 추가된 인수
    market_prices: np.ndarray,
    forwards: np.ndarray,
    strikes: np.ndarray,
    option_types: np.ndarray,
    T: float,
    optimization_target: str = 'variance', # 'variance' 또는 'price_abs'
) -> np.ndarray:
    """
    단일 만기 슬라이스에 대해 SVI 파라미터를 최적화합니다.
    최적화 목표(total variance 또는 option price)를 선택할 수 있습니다.
    """
    # 경계 조건은 기존과 동일하게 사용
    bounds = Bounds([-np.inf, 0.0001, -0.999, -np.inf, 0.0001], # a, m 하한 변경
                    [np.inf, np.inf, 0.999, np.inf, np.inf])  # 상한 변경 (L-BFGS-B는 무한대 지원)
    # bounds = Bounds([-1.0, 0.0001, -0.999, -5.0, 0.0001], [1.0, 10.0, 0.999, 5.0, 5.0])
    # 필요 시 기존 bounds 사용: Bounds([-1.0, 0.0001, -0.999, -5.0, 0.0001], [1.0, 10.0, 0.999, 5.0, 5.0])

    # --- 손실 함수 정의 ---
    def loss_variance(params: List[float]) -> float:
        """총 분산 MSE 손실 함수"""
        a, b, rho, m, sigma = params
        # 파라미터 유효성 기본 체크 (b, sigma > 0, |rho| < 1)
        if b <= 0 or sigma <= 0 or abs(rho) >= 1.0: return np.inf

        w_model = raw_svi_total_variance(k, a, b, rho, m, sigma)
        # 모델 분산이 음수가 되지 않도록 보장 (최소 0)
        w_model = np.maximum(w_model, 0.0)
        if not np.all(np.isfinite(w_model)): return np.inf
        return np.mean((w_model - w_market) ** 2)
    
    def loss_volatility(params: List[float]) -> float:
        """IV MSE 손실 함수"""
        a, b, rho, m, sigma = params
        # 파라미터 유효성 기본 체크
        if b <= 0 or sigma <= 0 or abs(rho) >= 1.0: return np.inf

        w_model = raw_svi_total_variance(k, a, b, rho, m, sigma)
        # 모델 분산 음수 방지 및 0 나누기 방지
        w_model = np.maximum(w_model, 1e-12)
        if not np.all(np.isfinite(w_model)): return np.inf
        
        # IV 계산
        iv_model = np.sqrt(w_model / T)
        iv_market = np.sqrt(w_market / T)
        return np.mean((iv_model - iv_market) ** 2)


    def loss_price_abs(params: List[float]) -> float:
        """절대 가격 MSE 손실 함수"""
        a, b, rho, m, sigma = params
        # 파라미터 유효성 기본 체크
        if b <= 0 or sigma <= 0 or abs(rho) >= 1.0: return np.inf

        w_model = raw_svi_total_variance(k, a, b, rho, m, sigma)
        # 모델 분산 음수 방지 및 0 나누기 방지
        w_model = np.maximum(w_model, 1e-12) # 가격 계산 위해 0보다 커야 함
        if not np.all(np.isfinite(w_model)): return np.inf

        # IV 계산
        if T <= 1e-8: return np.inf
        iv_model = np.sqrt(w_model / T)

        # 모델 가격 계산
        model_prices = np.array([
            black_scholes_price(f, kk, T, iv, opt_type)
            for f, kk, iv, opt_type in zip(forwards, strikes, iv_model, option_types)
        ])

        if not np.all(np.isfinite(model_prices)): return np.inf
        return np.mean((model_prices - market_prices) ** 2)
    
    def loss_price_rel(params: List[float]) -> float:
        """상대 가격 MSE 손실 함수"""
        a, b, rho, m, sigma = params
        # 파라미터 유효성 기본 체크
        if b <= 0 or sigma <= 0 or abs(rho) >= 1.0: return np.inf

        w_model = raw_svi_total_variance(k, a, b, rho, m, sigma)
        # 모델 분산 음수 방지 및 0 나누기 방지
        w_model = np.maximum(w_model, 1e-12)
        if not np.all(np.isfinite(w_model)): return np.inf

        # IV 계산
        if T <= 1e-8: return np.inf
        iv_model = np.sqrt(w_model / T)

        # 모델 가격 계산
        model_prices = np.array([
            black_scholes_price(f, kk, T, iv, opt_type)
            for f, kk, iv, opt_type in zip(forwards, strikes, iv_model, option_types)
        ])

        if not np.all(np.isfinite(model_prices)): return np.inf
        # 상대 오차 계산 (0 나누기 방지)
        relative_diff = (model_prices - market_prices) / np.maximum(market_prices, 1e-8)
        return np.mean(relative_diff ** 2)

    # --- 최적화 목표 선택 ---
    if optimization_target == 'variance':
        loss_function = loss_variance
    elif optimization_target == 'price_abs':
        loss_function = loss_price_abs
    elif optimization_target == 'price_rel':
        loss_function = loss_price_rel
    elif optimization_target == 'iv':
        loss_function = loss_volatility
    else:
        raise ValueError(f"Unknown optimization_target: {optimization_target}. Choose 'variance' or 'price_abs'.")

    # --- 최적화 실행 (L-BFGS-B 사용, fallback 제거) ---
    result = minimize(loss_function, initial_params, method='L-BFGS-B', bounds=bounds)

    # 최적화 실패 시 경고만 출력 (안정성 체크 보류)
    if not result.success:
        print(f"Warning: Raw SVI fitting may have failed for target '{optimization_target}' "
              f"on slice T={T:.4f}. Message: {result.message}")

    return result.x


# def fit_qr_svi_all_slices(df: pd.DataFrame, theta_map: Dict[pd.Timestamp, float], svi_square_root_func: Callable) -> Dict[pd.Timestamp, Dict[str, Any]]:
#     results = {}
#     for expiry, group in df.groupby('expiration'):
#         T = group['T'].iloc[0]
#         k_vals = group['k'].values
#         w_vals = group['total_variance'].values
#         w_sqrt = svi_square_root_func(k_vals, theta_map[expiry])
#         a_init = np.min(w_sqrt)
#         m_init = k_vals[np.argmin(np.abs(k_vals))]
#         init_params = [a_init, 0.1, -0.5, m_init, 0.1]
#         fitted_params = fit_raw_svi_slice(k_vals, w_vals, initial_params=init_params)
#         results[expiry] = {'T': T, 'params': fitted_params}
#     return results

# === 기존 함수 수정 ===
def fit_qr_svi_all_slices(
    df: pd.DataFrame,
    theta_map: Dict[pd.Timestamp, float],
    svi_square_root_func: Callable,
    optimization_target: str = 'variance' # 최적화 목표 인수 추가
) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """
    모든 만기 슬라이스에 대해 QR SVI 파라미터를 피팅합니다.
    최적화 목표(variance 또는 price_abs)를 선택할 수 있습니다.
    """
    results = {}
    for expiry, group in df.groupby('expiration'):
        T = group['T'].iloc[0]
        k_vals = group['k'].values
        w_vals = group['total_variance'].values # 분산 타겟용

        # 가격 타겟을 위한 데이터 추출
        market_prices = group['mark_price_usd'].values
        forwards = group['F'].values
        strikes = group['strike_price'].values
        option_types = group['type'].values

        # 초기값 계산 (기존 로직 유지)
        current_theta = theta_map.get(expiry, 1e-6) # 기본값 추가
        current_theta = max(current_theta, 1e-6)    # 0 방지
        try:
            w_sqrt = svi_square_root_func(k_vals, current_theta)
            if not np.all(np.isfinite(w_sqrt)) or np.any(w_sqrt < 0): raise ValueError("Invalid w_sqrt")
            a_init = min(np.min(w_sqrt), np.min(w_vals)) * 0.9
            m_init = k_vals[np.argmin(np.abs(k_vals))]
            # b, rho, sigma는 고정 초기값 사용 (단순화)
            b_init = 0.1
            rho_init = -0.7
            sigma_init = 0.1
        except: # 오류 시 기본값 사용
             a_init, b_init, rho_init, m_init, sigma_init = np.median(w_vals)*0.8, 0.1, -0.7, 0.0, 0.1

        init_params = [a_init, b_init, rho_init, m_init, sigma_init]

        # fit_raw_svi_slice 호출 시 추가 인수 전달
        fitted_params = fit_raw_svi_slice(
            k=k_vals,
            w_market=w_vals,
            initial_params=init_params,
            # 가격 관련 인수 전달
            market_prices=market_prices,
            forwards=forwards,
            strikes=strikes,
            option_types=option_types,
            T=T,
            # 최적화 목표 전달
            optimization_target=optimization_target
        )
        results[expiry] = {'T': T, 'params': fitted_params}
    return results


# -------------------------------
# 단계 3: Arbitrage 확인
# -------------------------------
def check_calendar_arbitrage(qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    expiries = sorted(qr_svi_fits.keys())
    warnings = []
    for i in range(len(expiries) - 1):
        exp1, exp2 = expiries[i], expiries[i + 1]
        w1 = raw_svi_total_variance(np.linspace(-0.5, 0.5, 50), *qr_svi_fits[exp1]['params'])
        w2 = raw_svi_total_variance(np.linspace(-0.5, 0.5, 50), *qr_svi_fits[exp2]['params'])
        if np.any(w2 < w1):
            warnings.append((exp1, exp2))
    return warnings


def check_butterfly_arbitrage(qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> List[pd.Timestamp]:
    arbitrage_slices = []
    k_vals = np.linspace(-2, 2, 100)
    for expiry, entry in qr_svi_fits.items():
        a, b, rho, m, sigma = entry['params']
        k_minus_m = k_vals - m
        sqrt_term = np.sqrt(k_minus_m**2 + sigma**2)
        w = a + b * (rho * k_minus_m + sqrt_term)
        dw = b * (rho + k_minus_m / sqrt_term)
        d2w = b * sigma**2 / (sqrt_term**3)
        # g = 1 - k_vals * dw / w + (dw**2 / 4) * (1 / w * (k_vals**2 / w - 4)) + d2w / 2
        g = (1 - 0.5 * k_vals * dw / w)**2 - 0.25 * dw**2 * (1 / w + 1 / 4) + d2w / 2
        if np.any(g < 0):
            arbitrage_slices.append(expiry)
    return arbitrage_slices


# -------------------------------
# 단계 4: Butterfly Arbitrage 시각화
# -------------------------------
def plot_svi_slice_with_density_check(expiry: pd.Timestamp, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    k_vals = np.linspace(-2, 2, 400)
    a, b, rho, m, sigma = qr_svi_fits[expiry]['params']
    k_m = k_vals - m
    sqrt_term = np.sqrt(k_m ** 2 + sigma ** 2)
    w = raw_svi_total_variance(k_vals, a, b, rho, m, sigma)
    dw = b * (rho + k_m / sqrt_term)
    d2w = b * sigma ** 2 / (sqrt_term ** 3)
    # g = 1 - k_vals * dw / w + (dw ** 2 / 4) * (1 / w * (k_vals ** 2 / w - 4)) + d2w / 2
    g = (1 - 0.5 * k_vals * dw / w)**2 - 0.25 * dw**2 * (1 / w + 1 / 4) + d2w / 2

    plt.figure(figsize=(10, 5))
    plt.title(f"SVI Total Variance and Density Check\nExpiry: {expiry.date()}")
    plt.plot(k_vals, w, label="Total Variance")
    plt.fill_between(k_vals, 0, w, where=(g < 0), color='red', alpha=0.3, label="g(k) < 0 (arbitrage)")
    plt.xlabel("log-moneyness (k)")
    plt.ylabel("Total Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------------
# 단계 5: Market IV 대 SVI IV 비교
# -------------------------------
def plot_market_vs_svi_iv(expiry: pd.Timestamp, df: pd.DataFrame, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> None:
    """
    주어진 expiry에 대해 market IV와 피팅된 SVI IV를 비교하고 플롯합니다.
    """
    if expiry not in qr_svi_fits:
        print("Expiry가 fit 결과에 없습니다.")
        return

    fit = qr_svi_fits[expiry]
    T = fit['T']
    a, b, rho, m, sigma = fit['params']

    df_slice = df[df['expiration'] == expiry]
    k = df_slice['k'].values
    iv_market = df_slice['mark_iv_decimal'].values
    w_svi = raw_svi_total_variance(k, a, b, rho, m, sigma)
    iv_svi = np.sqrt(w_svi / T)

    # 정렬된 순서로 선형 플롯
    sort_idx = np.argsort(k)
    k_sorted = k[sort_idx]
    iv_svi_sorted = iv_svi[sort_idx]

    plt.figure(figsize=(10, 5))
    plt.title(f"Market IV vs. SVI IV Expiry: {expiry.date()}")
    plt.scatter(k, iv_market, label="Market IV", alpha=0.7)
    plt.plot(k_sorted, iv_svi_sorted, color='red', label="SVI Fit", linewidth=2)
    plt.xlabel("log-moneyness (k)")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------
# 단계 6: Arbitrage Strikes 식별
# -------------------------------
def get_arbitrage_strikes(expiry: pd.Timestamp, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]], forward: float = 1.0, df_slice: Optional[pd.DataFrame] = None) -> List[float]:
    """
    g(k) < 0인 log-moneyness에 해당하는 strike 목록 반환
    forward 값은 기본적으로 1.0이며, 실제 underlying_price를 전달해야 정확
    """
    if expiry not in qr_svi_fits:
        return []

    a, b, rho, m, sigma = qr_svi_fits[expiry]['params']
    k_vals = np.linspace(-2, 2, 500)
    k_m = k_vals - m
    sqrt_term = np.sqrt(k_m**2 + sigma**2)
    w = a + b * (rho * k_m + sqrt_term)
    dw = b * (rho + k_m / sqrt_term)
    d2w = b * sigma**2 / (sqrt_term**3)

    # g = 1 - k_vals * dw / w + (dw**2 / 4) * (1 / w * (k_vals**2 / w - 4)) + d2w / 2
    g = (1 - 0.5 * k_vals * dw / w)**2 - 0.25 * dw**2 * (1 / w + 1 / 4) + d2w / 2
    k_arbs = k_vals[g < 0]

    strikes = forward * np.exp(k_arbs)
    # market-bound 필터링
    if df_slice is not None:
        strike_min = df_slice['strike_price'].min()
        strike_max = df_slice['strike_price'].max()
        strikes = [s for s in strikes if strike_min <= s <= strike_max]

    return np.round(strikes, 2).tolist()


# -------------------------------
# 단계 7: 만기별 Arbitrage Strikes 출력
# -------------------------------
def print_all_arbitrage_strikes(qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]], preprocessed_df: pd.DataFrame) -> None:
    """시장 범위 내의 butterfly arbitrage strikes를 출력합니다."""
    forward_by_expiry = preprocessed_df.groupby('expiration')['F'].mean().to_dict()
    for expiry in sorted(qr_svi_fits.keys()):
        forward = forward_by_expiry.get(expiry, 1.0)
        df_slice = preprocessed_df[preprocessed_df['expiration'] == expiry]
        strikes = get_arbitrage_strikes(expiry, qr_svi_fits, forward, df_slice)
        if strikes:
            print(f"\n[Butterfly Arbitrage Detected] {expiry.date()}:")
            print("Strikes:", strikes)


# -------------------------------
# 단계 8: Market IV 및 SVI IV 내보내기
# -------------------------------
def iv_comparison_to_dataframe(expiry: pd.Timestamp, df: pd.DataFrame, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    만기별로 SVI 대 market IV 비교 데이터를 DataFrame으로 반환합니다.
    """
    if expiry not in qr_svi_fits:
        print(f"Skip: {expiry} not in fit results")
        return None
    
    fit = qr_svi_fits[expiry]
    T = fit['T']
    a, b, rho, m, sigma = fit['params']
    
    df_slice = df[df['expiration'] == expiry].copy()
    k = df_slice['k'].values
    iv_market = df_slice['mark_iv_decimal'].values
    w_svi = raw_svi_total_variance(k, a, b, rho, m, sigma)
    iv_svi = np.sqrt(w_svi / T)
    
    result_df = df_slice[['expiration', 'strike_price', 'F', 'T', 'k']].copy()
    result_df['expiration'] = pd.to_datetime(result_df['expiration']).dt.strftime("%Y-%m-%d %H:%M:%S")
    result_df['iv_market'] = iv_market
    result_df['iv_svi'] = iv_svi
    result_df = result_df.sort_values(by='strike_price')
    return result_df


def export_all_iv_comparisons_to_dataframe(qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]], df: pd.DataFrame, directory: str = "./iv_exports", filename: str = "iv_comparison_all.csv") -> None:
    """
    전체 만기별 SVI 대 market IV 비교 데이터를 CSV로 저장합니다.
    """
    os.makedirs(directory, exist_ok=True)
    combined_data = []
    
    for expiry in qr_svi_fits:
        df_slice = iv_comparison_to_dataframe(expiry, df, qr_svi_fits)
        if df_slice is not None:
            combined_data.append(df_slice)
            
    if combined_data:
        all_df = pd.concat(combined_data, ignore_index=True)
        output_path = os.path.join(directory, filename)
        all_df.to_csv(output_path, index=False)
        print(f"[Saved] {output_path}")
    else:
        print("No data to save.")


def print_iv_error_metrics_by_expiry(df: pd.DataFrame, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> None:
    """
    각 만기별로 IV market 대 SVI fit 오차를 출력 (RMSE, MAE, Mean Error 포함)
    """
    print("\n[IV Market vs. SVI Fit Error Summary]")
    for expiry in qr_svi_fits:
        df_slice = df[df['expiration'] == expiry]
        if df_slice.empty:
            continue

        k = df_slice['k'].values
        iv_market = df_slice['mark_iv_decimal'].values
        T = qr_svi_fits[expiry]['T']
        a, b, rho, m, sigma = qr_svi_fits[expiry]['params']
        iv_svi = np.sqrt(raw_svi_total_variance(k, a, b, rho, m, sigma) / T)

        diff = iv_market - iv_svi
        mean_error = np.mean(diff)
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))

        # print(f"{expiry.date()} → RMSE: {rmse:.5f}, MAE: {mae:.5f}, Mean Error: {mean_error:.5f}")
        print(f"{expiry.date()} → RMSE: {rmse * 100:.3f}, MAE: {mae * 100:.3f}, Mean Error: {mean_error * 100:.3f}")
