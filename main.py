from svi_utils import *
from pricing import *

# 파일 경로는 사용 환경에 맞게 수정
OPTION_CSV_PATH = "./data/option_raw_data_2.csv"

# Step 0: 데이터 불러오기 및 전처리
df_raw = pd.read_csv(OPTION_CSV_PATH)
df = preprocess_option_data(df_raw)

# Step 1: Square-root SVI 초기화
svi_square = fit_square_root_svi(df)

# Step 2: QR SVI Fitting
qr_svi_fits = fit_qr_svi_all_slices(
    df,
    theta_map=svi_square['theta_by_expiry'],
    svi_square_root_func=svi_square['svi_func']
)

# Step 3: Arbitrage Check
calendar_arbs = check_calendar_arbitrage(qr_svi_fits)
butterfly_arbs = check_butterfly_arbitrage(qr_svi_fits)

# 결과 출력
print("\n[Calendar Arbitrage Detected]")
for pair in calendar_arbs:
    print(f"{pair[0]} -> {pair[1]}")

print("\n[Butterfly Arbitrage Detected]")
for exp in butterfly_arbs:
    print(exp)
    plot_svi_slice_with_density_check(pd.Timestamp(exp), qr_svi_fits)

print_all_arbitrage_strikes(qr_svi_fits, df)

# for key in qr_svi_fits.keys():
#     plot_market_vs_svi_iv(key, df, qr_svi_fits)

export_all_iv_comparisons_to_dataframe(qr_svi_fits, df, directory="./results")

print_iv_error_metrics_by_expiry(df, qr_svi_fits)

simulated_prices_df = simulate_option_prices_from_svi(df, qr_svi_fits)
simulated_prices_df.to_csv("./results/svi_price_comparison.csv", index=False)
print_price_error_metrics_by_expiry(df, qr_svi_fits)

local_vol_df = compute_local_vol_at_market_points(qr_svi_fits, df)
export_local_vol_with_iv(
    local_vol_df,
    df=df,
    qr_svi_fits=qr_svi_fits,
    output_path="./results/local_vol_market_only.csv"
)

