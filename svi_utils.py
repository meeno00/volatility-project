import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import os


# -------------------------------
# Step 0: Preprocessing
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

    # 로그 moneyness 및 total variance 계산
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
# Step 1: Square-root SVI Initialization
# -------------------------------
def square_root_svi_w(k, theta, rho, eta):
    phi = eta / (np.sqrt(theta) + 1e-8)  # numerical stability fix
    term1 = 1 + rho * phi * k
    term2 = np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
    return theta * (term1 + term2)


def fit_square_root_svi(df: pd.DataFrame):
    grouped = df.groupby('expiration')
    theta_by_expiry = {}
    slice_data = {}

    for expiry, group in grouped:
        group = group.copy()
        group['abs_k'] = np.abs(group['k'])
        atm_row = group.loc[group['abs_k'].idxmin()]
        theta_by_expiry[expiry] = atm_row['total_variance']
        slice_data[expiry] = group[['k', 'total_variance']]

    all_k, all_w, all_theta = [], [], []
    for expiry, data in slice_data.items():
        theta = theta_by_expiry[expiry]
        all_k.extend(data['k'].tolist())
        all_w.extend(data['total_variance'].tolist())
        all_theta.extend([theta] * len(data))

    all_k, all_w, all_theta = map(np.array, [all_k, all_w, all_theta])

    def svi_loss(params):
        rho, eta = params
        if not (-0.999 < rho < 0.999 and eta > 0): return np.inf
        w_model = square_root_svi_w(all_k, all_theta, rho, eta)
        if not np.all(np.isfinite(w_model)): return np.inf
        return np.mean((w_model - all_w) ** 2)

    result = minimize(svi_loss, [0.0, 0.5], bounds=[(-0.999, 0.999), (1e-4, 5.0)])
    fitted_rho, fitted_eta = result.x

    return {
        'rho': fitted_rho,
        'eta': fitted_eta,
        'theta_by_expiry': theta_by_expiry,
        'svi_func': lambda k, T: square_root_svi_w(k, T, fitted_rho, fitted_eta)
    }


# -------------------------------
# Step 2: QR SVI Fitting
# -------------------------------
def raw_svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def fit_raw_svi_slice(k, w, initial_params):
    bounds = Bounds([-1.0, 0.0001, -0.999, -5.0, 0.0001], [1.0, 10.0, 0.999, 5.0, 5.0])

    def loss(params):
        a, b, rho, m, sigma = params
        model = raw_svi_total_variance(k, a, b, rho, m, sigma)
        return np.mean((model - w) ** 2)
    return minimize(loss, initial_params, bounds=bounds, method='L-BFGS-B').x


def fit_qr_svi_all_slices(df, theta_map, svi_square_root_func):
    results = {}
    for expiry, group in df.groupby('expiration'):
        T = group['T'].iloc[0]
        k_vals = group['k'].values
        w_vals = group['total_variance'].values
        w_sqrt = svi_square_root_func(k_vals, theta_map[expiry])
        a_init = np.min(w_sqrt)
        m_init = k_vals[np.argmin(np.abs(k_vals))]
        init_params = [a_init, 0.1, -0.5, m_init, 0.1]
        fitted_params = fit_raw_svi_slice(k_vals, w_vals, initial_params=init_params)
        results[expiry] = {'T': T, 'params': fitted_params}
    return results


# -------------------------------
# Step 3: Arbitrage Check
# -------------------------------
def check_calendar_arbitrage(qr_svi_fits):
    expiries = sorted(qr_svi_fits.keys())
    warnings = []
    for i in range(len(expiries) - 1):
        exp1, exp2 = expiries[i], expiries[i + 1]
        w1 = raw_svi_total_variance(np.linspace(-0.5, 0.5, 50), *qr_svi_fits[exp1]['params'])
        w2 = raw_svi_total_variance(np.linspace(-0.5, 0.5, 50), *qr_svi_fits[exp2]['params'])
        if np.any(w2 < w1):
            warnings.append((exp1, exp2))
    return warnings


def check_butterfly_arbitrage(qr_svi_fits):
    arbitrage_slices = []
    k_vals = np.linspace(-2, 2, 100)
    for expiry, entry in qr_svi_fits.items():
        a, b, rho, m, sigma = entry['params']
        k_minus_m = k_vals - m
        sqrt_term = np.sqrt(k_minus_m**2 + sigma**2)
        w = a + b * (rho * k_minus_m + sqrt_term)
        dw = b * (rho + k_minus_m / sqrt_term)
        d2w = b * sigma**2 / (sqrt_term**3)
        g = 1 - k_vals * dw / w + (dw**2 / 4) * (1 / w * (k_vals**2 / w - 4)) + d2w / 2
        if np.any(g < 0):
            arbitrage_slices.append(expiry)
    return arbitrage_slices


# -------------------------------
# Step 4: Visualization of Butterfly Arbitrage
# -------------------------------
def plot_svi_slice_with_density_check(expiry, qr_svi_fits):
    import matplotlib.pyplot as plt
    k_vals = np.linspace(-2, 2, 400)
    a, b, rho, m, sigma = qr_svi_fits[expiry]['params']
    k_m = k_vals - m
    sqrt_term = np.sqrt(k_m ** 2 + sigma ** 2)
    w = raw_svi_total_variance(k_vals, a, b, rho, m, sigma)
    dw = b * (rho + k_m / sqrt_term)
    d2w = b * sigma ** 2 / (sqrt_term ** 3)
    g = 1 - k_vals * dw / w + (dw ** 2 / 4) * (1 / w * (k_vals ** 2 / w - 4)) + d2w / 2

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
# Step 5: Compare Market IV vs. SVI IV
# -------------------------------
def plot_market_vs_svi_iv(expiry, df, qr_svi_fits):
    """
    Compare and plot market IV vs. fitted SVI IV for a given expiry.
    """
    if expiry not in qr_svi_fits:
        print("Expiry not in fit result.")
        return

    fit = qr_svi_fits[expiry]
    T = fit['T']
    a, b, rho, m, sigma = fit['params']

    df_slice = df[df['expiration'] == expiry]
    k = df_slice['k'].values
    iv_market = df_slice['mark_iv_decimal'].values
    w_svi = raw_svi_total_variance(k, a, b, rho, m, sigma)
    iv_svi = np.sqrt(w_svi / T)

    # 정렬된 순서로 선형 plot
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
# Step 6: Identify Arbitrage Strikes
# -------------------------------
def get_arbitrage_strikes(expiry, qr_svi_fits, forward=1.0, df_slice=None):
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

    g = 1 - k_vals * dw / w + (dw**2 / 4) * (1 / w * (k_vals**2 / w - 4)) + d2w / 2
    k_arbs = k_vals[g < 0]

    # strike = F * exp(k)
    strikes = forward * np.exp(k_arbs)
    # market-bound filtering
    if df_slice is not None:
        strike_min = df_slice['strike_price'].min()
        strike_max = df_slice['strike_price'].max()
        strikes = [s for s in strikes if strike_min <= s <= strike_max]

    return np.round(strikes, 2).tolist()


# -------------------------------
# Step 7: Print Arbitrage Strikes by Expiry
# -------------------------------
def print_all_arbitrage_strikes(qr_svi_fits, preprocessed_df):
    """Print butterfly arbitrage strikes within market range."""
    forward_by_expiry = preprocessed_df.groupby('expiration')['F'].mean().to_dict()
    for expiry in sorted(qr_svi_fits.keys()):
        forward = forward_by_expiry.get(expiry, 1.0)
        df_slice = preprocessed_df[preprocessed_df['expiration'] == expiry]
        strikes = get_arbitrage_strikes(expiry, qr_svi_fits, forward, df_slice)
        if strikes:
            print(f"\n[Butterfly Arbitrage Detected] {expiry.date()}:")
            print("Strikes:", strikes)


# -------------------------------
# Step 8: Export Market IV and SVI IV
# -------------------------------
def export_iv_comparison_to_csv(expiry, df, qr_svi_fits, output_path):
    if expiry not in qr_svi_fits:
        print(f"Skip: {expiry} not in fit results")
        return

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
    result_df.to_csv(output_path, index=False)


def export_all_iv_comparisons(qr_svi_fits, df, directory="./iv_exports"):
    """
    전체 만기별로 SVI vs 시장 IV 비교 데이터를 CSV로 저장한다.
    """
    os.makedirs(directory, exist_ok=True)

    for expiry in qr_svi_fits:
        file_name = f"iv_comparison_{expiry.date()}.csv"
        path = os.path.join(directory, file_name)
        export_iv_comparison_to_csv(expiry, df, qr_svi_fits, path)
        print(f"[Saved] {path}")


def print_iv_error_metrics_by_expiry(df, qr_svi_fits):
    """
    각 만기별로 IV market vs SVI fit 오차를 출력 (RMSE, MAE, Mean Error 포함)
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
