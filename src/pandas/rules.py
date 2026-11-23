"""
Shared cleaning & feature-engineering rules – HMDA 2024 (Group 6)
Dùng chung cho Pandas và PySpark. Chỉ chứa hằng số / bảng ánh xạ.
"""

# ---------- 1. Mã thiếu / đặc biệt ----------
MISSING_STRINGS = ["", "NA", "Exempt"]   # -> missing ở mọi cột
EXEMPT_CODE = "1111"                      # -> missing, chỉ ở CODE_COLS
AGE_NA_CODE = "8888"                      # -> missing, chỉ ở AGE_COLS

CODE_COLS = ["action_taken", "loan_type", "loan_purpose", "lien_status",
             "occupancy_type", "construction_method",
             "denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4"]
AGE_COLS = ["applicant_age"]

# ---------- 2. Kiểu dữ liệu ----------
NUMERIC_COLS = ["income", "loan_amount", "combined_loan_to_value_ratio", "property_value",
                "tract_population", "tract_minority_population_percent",
                "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
                "tract_owner_occupied_units", "tract_one_to_four_family_homes",
                "tract_median_age_of_housing_units"]
# Các cột còn lại giữ dạng chuỗi/category (mã, ID: state_code, county_code, census_tract, lei)

# ---------- 3. Giá trị không hợp lệ: set missing + gắn cờ, KHÔNG xoá dòng ----------
IMPLAUSIBLE = {
    "income": ("le", 0),                          # income <= 0
    "combined_loan_to_value_ratio": ("gt", 250),  # CLTV > 250%
}

# ---------- 4. Dòng trùng ----------
DUPLICATE_ACTION = "flag"   # gắn cờ, không xoá (HMDA không có mã hồ sơ)

# ---------- 5. Outcome & samples ----------
ORIGINATED = "1"
DENIED = "3"
S1_CODES = ["1", "3"]       # originated & denied
S2_CODES = ["3"]            # denied only
# Denial rate = denied / (originated + denied)

# ---------- 6. Feature engineering ----------
INCOME_UNIT = 1000          # income tính theo nghìn USD; loan_amount theo USD

DTI_BAND_MAP = {"<20%": "<20%", "20%-<30%": "20-<30%", "30%-<36%": "30-<36%",
                "50%-60%": "50-60%", ">60%": ">60%"}
DTI_BAND_MAP.update({str(v): "36-43%" for v in range(36, 44)})
DTI_BAND_MAP.update({str(v): "44-49%" for v in range(44, 50)})
DTI_BAND_ORDER = ["<20%", "20-<30%", "30-<36%", "36-43%", "44-49%", "50-60%", ">60%"]

# CLTV: khoảng ĐÓNG BÊN PHẢI (a, b]  -> PySpark dùng when(col <= b), KHÔNG dùng Bucketizer [a, b)
CLTV_UPPER = [60, 80, 90, 95, 100, 250]
CLTV_BAND_ORDER = ["<=60", "60-80", "80-90", "90-95", "95-100", "100-250"]

# ---------- 7. Leakage guard (không bao giờ đưa vào model) ----------
LEAKAGE_COLS = ["denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4",
                "interest_rate", "rate_spread", "total_loan_costs", "origination_charges",
                "discount_points", "lender_credits", "purchaser_type", "hoepa_status",
                "total_points_and_fees"]
