import numpy as np
from scipy.stats import norm
import pandas as pd
from typing import Dict, List, Any

def black_scholes_price(F: float, K: float, T: float, sigma: float, option_type: str) -> float:
    """
    Forward 기반 Black-Scholes 옵션 가격 계산 (할인율 1 가정)

    Parameters:
    - F : forward price
    - K : strike price
    - T : time to maturity (in years)
    - sigma : implied volatility
    - option_type : 'call' or 'put'

    Returns:
    - option price
    """
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) if option_type == 'call' else max(K - F, 0)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    elif option_type == 'put':
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def simulate_option_prices_from_svi(df: pd.DataFrame, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> pd.DataFrame:
    """
    SVI IV로부터 Black-Scholes 옵션 가격을 재계산하여 mark_price와 비교

    Returns:
    - DataFrame with [expiration, strike_price, option_type, T, iv_svi, price_svi, price_market]
    """
    results = []

    for expiry, fit in qr_svi_fits.items():
        a, b, rho, m, sigma = fit['params']
        T = fit['T']

        df_slice = df[df['expiration'] == expiry].copy()
        if df_slice.empty:
            continue

        for _, row in df_slice.iterrows():
            k = row['k']
            K = row['strike_price']
            F = row['F']
            option_type = row['type']
            market_price = row['mark_price_usd']

            w_svi = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
            iv_svi = np.sqrt(w_svi / T)

            model_price = black_scholes_price(F, K, T, iv_svi, option_type)

            results.append({
                'expiration': expiry,
                'strike_price': K,
                'option_type': option_type,
                'T': T,
                'F': F,
                'iv_svi': iv_svi,
                'price_svi': model_price,
                'price_market': market_price,
                'price_diff': market_price - model_price
            })

    return pd.DataFrame(results)


def print_price_error_metrics_by_expiry(df: pd.DataFrame, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]]) -> None:
    """
    각 만기별로 SVI 모델 가격 vs 시장 가격의 RMSE, MAE, Mean Error를 출력
    """
    print("\n[Option Price Error Summary (SVI vs Market)]")

    for expiry in qr_svi_fits:
        df_slice = df[df['expiration'] == expiry]
        if df_slice.empty:
            continue

        a, b, rho, m, sigma = qr_svi_fits[expiry]['params']
        T = qr_svi_fits[expiry]['T']

        F = df_slice['F'].values
        K = df_slice['strike_price'].values
        opt_type = df_slice['type'].values
        market_price = df_slice['mark_price_usd'].values
        k = df_slice['k'].values

        w_svi = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        iv_svi = np.sqrt(w_svi / T)

        model_price = np.array([
            black_scholes_price(f, k_, T, iv_, t)
            for f, k_, iv_, t in zip(F, K, iv_svi, opt_type)
        ])

        diff = market_price - model_price
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        mean_error = np.mean(diff)

        # ✅ Mean Relative Error (%)
        relative_error = np.abs(diff) / (market_price + 1e-8)  # 작은 값 방지용 epsilon
        mean_relative_error = np.mean(relative_error) * 100  # 퍼센트로 환산

        print(f"{expiry.date()} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, Mean Error: {mean_error:.2f}, Mean RelErr: {mean_relative_error:.2f}%")


def compute_local_vol_from_svi_linear_with_extrapolation(qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]], k_grid: np.ndarray, T_eval_grid: np.ndarray) -> pd.DataFrame:
    """
    Dupire local volatility 계산 (∂w/∂T: linear + forward/backward diff 포함)
    Returns:
        DataFrame with columns: ['k', 'T', 'local_vol']
    """
    expiries = sorted(qr_svi_fits.keys())
    T_list = [qr_svi_fits[exp]['T'] for exp in expiries]

    def w_svi(k: float, params: List[float]) -> float:
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    local_vol_data = []

    for T in T_eval_grid:
        # T와 가장 가까운 T_list 인덱스를 찾음
        idx = np.searchsorted(T_list, T)

        # T가 T_list 범위 내
        if 0 < idx < len(T_list) - 1:
            # 중앙차분
            T1, T2 = T_list[idx - 1], T_list[idx + 1]
            p1 = qr_svi_fits[expiries[idx - 1]]['params']
            p2 = qr_svi_fits[expiries[idx + 1]]['params']
            dw_mode = "central"
        elif idx == 0:
            # 앞단 extrapolation → forward diff
            T1, T2 = T_list[0], T_list[1]
            p1 = qr_svi_fits[expiries[0]]['params']
            p2 = qr_svi_fits[expiries[1]]['params']
            dw_mode = "forward"
        elif idx >= len(T_list) - 1:
            # 뒷단 extrapolation → backward diff
            T1, T2 = T_list[-2], T_list[-1]
            p1 = qr_svi_fits[expiries[-2]]['params']
            p2 = qr_svi_fits[expiries[-1]]['params']
            dw_mode = "backward"
        else:
            # 잘못된 경우 생략
            print(f"[Skipped T: {T:.4f}] → np.searchsorted(T_list, T) returned invalid idx = {idx}")
            continue

        for k in k_grid:
            # Total variance
            w1 = w_svi(k, p1)
            w2 = w_svi(k, p2)

            # ∂w/∂T 선형근사
            dw_dT = (w2 - w1) / (T2 - T1)

            # 미분 근사 기준 파라미터 선택
            if dw_mode == "central":
                params = qr_svi_fits[expiries[idx]]['params']
                w_fixed = w_svi(k, params)
            elif dw_mode == "forward":
                params = p1
                w_fixed = w1
            else:  # "backward"
                params = p2
                w_fixed = w2

            a, b, rho, m, sigma = params
            km = k - m
            sqrt_term = np.sqrt(km**2 + sigma**2 + 1e-12)

            dw_dk = b * (rho + km / sqrt_term)
            d2w_dk2 = b * sigma**2 / (sqrt_term**3)

            denom = (
                1 - k * dw_dk / w_fixed +
                0.25 * (dw_dk ** 2) * (-0.25 - 1 / w_fixed + (k**2 / w_fixed**2)) +
                0.5 * d2w_dk2
            )

            if denom <= 0 or dw_dT <= 0:
                reason = "denominator ≤ 0" if denom <= 0 else "dw_dT ≤ 0"
                print(f"[Skipped] T={T:.4f}, k={k:.4f} → {reason}")
                continue

            local_var = dw_dT / denom
            local_vol = np.sqrt(local_var)

            local_vol_data.append({
                'k': k,
                'T': T,
                'local_vol': local_vol
            })

    return pd.DataFrame(local_vol_data)


def compute_local_vol_at_market_points(qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]], df: pd.DataFrame) -> pd.DataFrame:
    """
    시장 데이터에 실제 존재하는 (T, k) 조합에서만 Dupire local volatility 계산
    Returns:
        DataFrame with columns: ['expiration', 'strike_price', 'k', 'T', 'local_vol']
    """
    expiries = sorted(qr_svi_fits.keys())
    T_list = [qr_svi_fits[exp]['T'] for exp in expiries]

    def w_svi(k: float, params: List[float]) -> float:
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    local_vol_data = []

    for expiry in expiries:
        T = qr_svi_fits[expiry]['T']
        df_slice = df[df['expiration'] == expiry]
        k_list = df_slice['k'].values
        strikes = df_slice['strike_price'].values

        # ∂w/∂T 계산을 위한 T 구간 결정
        idx = np.searchsorted(T_list, T)
        if 0 < idx < len(T_list) - 1:
            T1, T2 = T_list[idx - 1], T_list[idx + 1]
            p1 = qr_svi_fits[expiries[idx - 1]]['params']
            p2 = qr_svi_fits[expiries[idx + 1]]['params']
            dw_mode = "central"
        elif idx == 0:
            T1, T2 = T_list[0], T_list[1]
            p1 = qr_svi_fits[expiries[0]]['params']
            p2 = qr_svi_fits[expiries[1]]['params']
            dw_mode = "forward"
        elif idx >= len(T_list) - 1:
            T1, T2 = T_list[-2], T_list[-1]
            p1 = qr_svi_fits[expiries[-2]]['params']
            p2 = qr_svi_fits[expiries[-1]]['params']
            dw_mode = "backward"
        else:
            print(f"[Skipped T: {T:.4f}] → np.searchsorted(T_list, T) returned invalid idx = {idx}")
            continue

        for k, strike in zip(k_list, strikes):
            w1 = w_svi(k, p1)
            w2 = w_svi(k, p2)
            dw_dT = (w2 - w1) / (T2 - T1)

            # 도함수 기준 params 선택
            if dw_mode == "central":
                params = qr_svi_fits[expiry]['params']
                w_fixed = w_svi(k, params)
            elif dw_mode == "forward":
                params = p1
                w_fixed = w1
            else:
                params = p2
                w_fixed = w2

            a, b, rho, m, sigma = params
            km = k - m
            sqrt_term = np.sqrt(km ** 2 + sigma ** 2 + 1e-12)

            dw_dk = b * (rho + km / sqrt_term)
            d2w_dk2 = b * sigma**2 / (sqrt_term**3)

            denom = (
                1 - k * dw_dk / w_fixed +
                0.25 * (dw_dk ** 2) * (-0.25 - 1 / w_fixed + (k**2 / w_fixed**2)) +
                0.5 * d2w_dk2
            )

            if denom <= 0 or dw_dT <= 0:
                reason = "denominator ≤ 0" if denom <= 0 else "dw_dT ≤ 0"
                print(f"[Skipped] T={T:.4f}, k={k:.4f} → {reason}")
                continue

            local_var = dw_dT / denom
            local_vol = np.sqrt(local_var)

            local_vol_data.append({
                'expiration': expiry,
                'strike_price': strike,
                'k': k,
                'T': T,
                'local_vol': local_vol
            })

    return pd.DataFrame(local_vol_data)


def export_local_vol_with_iv(local_vol_df: pd.DataFrame, df: pd.DataFrame, qr_svi_fits: Dict[pd.Timestamp, Dict[str, Any]], output_path: str) -> None:
    """
    local_vol_df에 시장 IV와 SVI IV를 추가하여 정제 후 CSV 저장
    """
    enriched_df = local_vol_df.copy()

    # expiration 포맷 정리
    enriched_df['expiration'] = pd.to_datetime(enriched_df['expiration']).dt.strftime("%Y-%m-%d %H:%M")

    # 시장 IV 추가: df 기준으로 매칭
    df_lookup = df[['expiration', 'strike_price', 'mark_iv_decimal']].copy()
    df_lookup['expiration'] = pd.to_datetime(df_lookup['expiration']).dt.strftime("%Y-%m-%d %H:%M")
    enriched_df = enriched_df.merge(df_lookup, on=['expiration', 'strike_price'], how='left')
    enriched_df = enriched_df.rename(columns={'mark_iv_decimal': 'iv_market'})

    # # SVI IV 추가: sqrt(w(k) / T)
    # def svi_iv(row):
    #     expiry = pd.to_datetime(row['expiration'])
    #     k = row['k']
    #     T = row['T']
    #     if expiry in qr_svi_fits:
    #         a, b, rho, m, sigma = qr_svi_fits[expiry]['params']
    #         w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    #         return np.sqrt(w / T)
    #     return np.nan
    #
    # enriched_df['iv_svi'] = enriched_df.apply(svi_iv, axis=1)

    # 정렬: expiration → strike_price
    enriched_df = enriched_df.sort_values(by=['expiration', 'strike_price'])

    # 저장
    enriched_df.to_csv(output_path, index=False)



