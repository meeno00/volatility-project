import time
from svi_utils import *
from pricing import *

# 파일 경로는 사용 환경에 맞게 수정
OPTION_CSV_PATH = "./data/option_raw_data.csv"
RESULTS_DIR = "./results"

# === 최적화 방법 설정 ===
# 'variance': 총 분산 MSE 최소화 (기존 방식)
# 'price_abs': 절대 옵션 가격 MSE 최소화
# 'price_rel': 상대 옵션 가격 MSE 최소화
# 'iv': IV MSE 최소화
OPTIMIZATION_TARGET = 'iv' # <--- 여기를 'variance', 'price_abs', 'price_rel'로 변경하여 최적화 방법을 선택

# --- 실행 ---
start_time = time.time()
os.makedirs(RESULTS_DIR, exist_ok=True) # 결과 디렉토리 생성
price_comparison_path = os.path.join(RESULTS_DIR, "svi_price_comparison.csv")
local_vol_path = os.path.join(RESULTS_DIR, "local_vol_market_only.csv")


# Step 0: 데이터 불러오기 및 전처리
print("Step 0: Load and preprocess data...")
df_raw = pd.read_csv(OPTION_CSV_PATH)
df = preprocess_option_data(df_raw)
print(f"Preprocessing complete. Data shape: {df.shape}")


# Step 1: Square-root SVI 초기화
print("\nStep 1: Fit Square-root SVI for initialization...")
svi_square = fit_square_root_svi(df)
print(f"Rho: {svi_square['rho']:.4f}, Eta: {svi_square['eta']:.4f}")


# Step 2: QR SVI Fitting
print(f"\nStep 2: Fit Raw SVI slices (Target: {OPTIMIZATION_TARGET})...")
qr_svi_fits = fit_qr_svi_all_slices(
    df,
    theta_map=svi_square['theta_by_expiry'],
    svi_square_root_func=svi_square['svi_func'],
    optimization_target=OPTIMIZATION_TARGET # 최적화 목표 전달
)
print(f"Fitting complete for {len(qr_svi_fits)} expiries.")


# Step 3: Arbitrage Check
print("\nStep 3: Check for Arbitrage...")
calendar_arbs = check_calendar_arbitrage(qr_svi_fits)
butterfly_arbs = check_butterfly_arbitrage(qr_svi_fits)

# 결과 출력
if calendar_arbs:
    print("\n[Calendar Arbitrage Detected]")
    for pair in calendar_arbs:
        print(f"{pair[0]} -> {pair[1]}")

if butterfly_arbs:
    print("\n[Butterfly Arbitrage Detected]")
    for exp in butterfly_arbs:
        print(exp)
        plot_svi_slice_with_density_check(pd.Timestamp(exp), qr_svi_fits)

print_all_arbitrage_strikes(qr_svi_fits, df)

# for key in qr_svi_fits.keys():
#     plot_market_vs_svi_iv(key, df, qr_svi_fits)

export_all_iv_comparisons_to_dataframe(qr_svi_fits, df, directory=RESULTS_DIR)
print_iv_error_metrics_by_expiry(df, qr_svi_fits)

simulated_prices_df = simulate_option_prices_from_svi(df, qr_svi_fits)
simulated_prices_df.to_csv(price_comparison_path, index=False)
print_price_error_metrics_by_expiry(df, qr_svi_fits)

local_vol_df = compute_local_vol_at_market_points(qr_svi_fits, df)
export_local_vol_with_iv(
    local_vol_df,
    df=df,
    qr_svi_fits=qr_svi_fits,
    output_path=local_vol_path
)

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
