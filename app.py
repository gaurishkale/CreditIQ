import streamlit as st
import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np

# ── Load raw booster (bypasses sklearn wrapper entirely) ───────────────────────
@st.cache_resource
def load_model():
    booster        = lgb.Booster(model_file="booster_only.txt")
    model_features = booster.feature_name()   # get features from booster directly
    pandas_cats    = booster.dump_model().get('pandas_categorical', [])
    return booster, model_features, pandas_cats

booster, model_features, pandas_cats = load_model()

# Map column names to their exact category lists (from booster training)
CAT_COLS = ['home_ownership', 'verification_status', 'purpose',
            'addr_state', 'initial_list_status', 'application_type']
CAT_DEFINITIONS = dict(zip(CAT_COLS, pandas_cats))

EMP_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4,  '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8,  '9 years': 9, '10+ years': 10
}
GRADE_MAP    = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
SUBGRADE_MAP = {f'{g}{i}': gv * 10 - i
                for g, gv in GRADE_MAP.items() for i in range(1, 6)}

def get_risk_details(prob):
    score = int(850 - prob * 550)
    if score >= 750:   grade = "A — Excellent"
    elif score >= 700: grade = "B — Good"
    elif score >= 650: grade = "C — Fair"
    elif score >= 600: grade = "D — Poor"
    elif score >= 550: grade = "E — Very Poor"
    else:              grade = "F — High Risk"
    if score >= 650:   decision = "✅ APPROVED"
    elif score >= 580: decision = "⚠️ MANUAL REVIEW"
    else:              decision = "❌ REJECTED"
    return score, grade, decision

def build_input(f: dict) -> pd.DataFrame:
    loan        = f['loan_amnt']
    rate        = f['int_rate']
    term        = f['term']
    inc         = f['annual_inc']
    revbal      = f['revol_bal']
    installment = round((loan * (rate / 1200)) /
                        (1 - (1 + rate / 1200) ** -term), 2)

    row = {
        "loan_amnt":            loan,
        "term":                 term,
        "int_rate":             rate,
        "installment":          installment,
        "emp_length":           EMP_MAP.get(f['emp_length'], 0),
        "home_ownership":       f['home_ownership'],
        "annual_inc":           inc,
        "verification_status":  f['verification_status'],
        "purpose":              f['purpose'],
        "addr_state":           f['addr_state'],
        "dti":                  f['dti'],
        "delinq_2yrs":          f['delinq_2yrs'],
        "inq_last_6mths":       f['inq_last_6mths'],
        "mths_since_last_delinq":       0,
        "mths_since_last_record":       0,
        "open_acc":             f['open_acc'],
        "pub_rec":              f['pub_rec'],
        "revol_bal":            revbal,
        "revol_util":           f['revol_util'],
        "total_acc":            f['total_acc'],
        "initial_list_status":  "w",
        "last_fico_range_high": f['fico_high'],
        "last_fico_range_low":  f['fico_low'],
        "collections_12_mths_ex_med":           0,
        "mths_since_last_major_derog":          0,
        "application_type":     f['application_type'],
        "annual_inc_joint":     0,
        "dti_joint":            0,
        "acc_now_delinq":       0,
        "tot_coll_amt":         0,
        "tot_cur_bal":          0,
        "open_acc_6m":          0,
        "open_act_il":          0,
        "open_il_12m":          0,
        "open_il_24m":          0,
        "mths_since_rcnt_il":   0,
        "total_bal_il":         0,
        "il_util":              0,
        "open_rv_12m":          0,
        "open_rv_24m":          0,
        "max_bal_bc":           0,
        "all_util":             f['revol_util'],
        "total_rev_hi_lim":     0,
        "inq_fi":               0,
        "total_cu_tl":          0,
        "inq_last_12m":         0,
        "acc_open_past_24mths": 0,
        "avg_cur_bal":          0,
        "bc_open_to_buy":       0,
        "bc_util":              0,
        "chargeoff_within_12_mths":     0,
        "delinq_amnt":          0,
        "mo_sin_old_il_acct":   0,
        "mo_sin_old_rev_tl_op": 0,
        "mo_sin_rcnt_rev_tl_op":0,
        "mo_sin_rcnt_tl":       0,
        "mort_acc":             f.get('mort_acc', 0),
        "mths_since_recent_bc":          0,
        "mths_since_recent_bc_dlq":      0,
        "mths_since_recent_inq":         0,
        "mths_since_recent_revol_delinq":0,
        "num_accts_ever_120_pd":0,
        "num_actv_bc_tl":       0,
        "num_actv_rev_tl":      0,
        "num_bc_sats":          0,
        "num_bc_tl":            0,
        "num_il_tl":            0,
        "num_op_rev_tl":        0,
        "num_rev_accts":        0,
        "num_rev_tl_bal_gt_0":  0,
        "num_sats":             f['open_acc'],
        "num_tl_120dpd_2m":     0,
        "num_tl_30dpd":         0,
        "num_tl_90g_dpd_24m":   0,
        "num_tl_op_past_12m":   0,
        "pct_tl_nvr_dlq":       90,
        "percent_bc_gt_75":     0,
        "pub_rec_bankruptcies": f.get('pub_rec_bankruptcies', 0),
        "tax_liens":            0,
        "tot_hi_cred_lim":      0,
        "total_bal_ex_mort":    0,
        "total_bc_limit":       0,
        "revol_bal_joint":      0,
        "sec_app_fico_range_low":              0,
        "sec_app_fico_range_high":             0,
        "sec_app_inq_last_6mths":              0,
        "sec_app_mort_acc":                    0,
        "sec_app_open_acc":                    0,
        "sec_app_revol_util":                  0,
        "sec_app_open_act_il":                 0,
        "sec_app_num_rev_accts":               0,
        "sec_app_chargeoff_within_12_mths":    0,
        "sec_app_collections_12_mths_ex_med":  0,
        "sec_app_mths_since_last_major_derog": 0,
        "orig_projected_additional_accrued_interest": 0,
        "credit_history_months":f.get('credit_history_months', 120),
        "issue_year":           2024,
        "issue_month":          1,
        "grade_num":            GRADE_MAP.get(f['grade'], 4),
        "sub_grade_num":        SUBGRADE_MAP.get(f['sub_grade'], 40),
        "fico_avg":             (f['fico_low'] + f['fico_high']) / 2,
        "loan_to_income":       loan / (inc + 1),
        "installment_to_income":installment / (inc / 12 + 1),
        "revol_bal_to_income":  revbal / (inc + 1),
        "credit_util_ratio":    f['revol_util'] / 100,
        "delinq_to_open_acc":   f['delinq_2yrs'] / (f['open_acc'] + 1),
        "inq_to_open_acc":      f['inq_last_6mths'] / (f['open_acc'] + 1),
        "pub_rec_flag":         int(f['pub_rec'] > 0),
        "bankruptcy_flag":      int(f.get('pub_rec_bankruptcies', 0) > 0),
        "high_dti_flag":        int(f['dti'] > 35),
        "high_int_rate_flag":   int(rate > 20),
        "log_annual_inc":       np.log1p(inc),
        "log_loan_amnt":        np.log1p(loan),
        "log_revol_bal":        np.log1p(revbal),
    }

    df = pd.DataFrame([row])

    # Fill any missing columns with 0
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Exact column order
    df = df[model_features]

    # Cast categoricals with exact training category lists
    for col, categories in CAT_DEFINITIONS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Loan Underwriting", layout="wide", page_icon="💳")
st.title("💳 AI-Powered Loan Underwriting & Risk Scoring")
st.markdown("Fill in the applicant's details and click **Evaluate Risk**.")
st.divider()

tab1, tab2 = st.tabs(["📋 Application Form", "ℹ️ About"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💰 Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)",   1000, 100000, 10000, step=500)
        int_rate  = st.slider("Interest Rate (%)",        5.0,   30.0,  12.0, step=0.1)
        term      = st.selectbox("Term (months)",        [36, 60])
        purpose   = st.selectbox("Loan Purpose",         CAT_DEFINITIONS['purpose'])
        grade     = st.selectbox("Loan Grade",           ['A','B','C','D','E','F','G'])
        sub_grade = st.selectbox("Sub Grade",
                        [f'{g}{i}' for g in ['A','B','C','D','E','F','G']
                         for i in range(1, 6)])

    with col2:
        st.subheader("👤 Applicant Profile")
        annual_inc       = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        emp_length       = st.selectbox("Employment Length",    list(EMP_MAP.keys()))
        home_own         = st.selectbox("Home Ownership",       CAT_DEFINITIONS['home_ownership'])
        verification     = st.selectbox("Income Verification",  CAT_DEFINITIONS['verification_status'])
        application_type = st.selectbox("Application Type",     CAT_DEFINITIONS['application_type'])
        addr_state       = st.selectbox("State",                CAT_DEFINITIONS['addr_state'])

    with col3:
        st.subheader("📊 Credit Profile")
        fico_low       = st.number_input("FICO Score Low",       300, 850, 680)
        fico_high      = st.number_input("FICO Score High",      300, 850, 700)
        dti            = st.slider("Debt-to-Income Ratio (%)",   0.0, 50.0, 15.0, step=0.1)
        revol_util     = st.slider("Revolving Utilization (%)",  0.0,100.0, 40.0, step=0.1)
        revol_bal      = st.number_input("Revolving Balance ($)", 0, 500000, 5000, step=100)
        open_acc       = st.number_input("Open Accounts",         0,  50,   8)
        total_acc      = st.number_input("Total Accounts",        0, 100,  20)
        delinq_2yrs    = st.number_input("Delinquencies (2yr)",   0,  20,   0)
        pub_rec        = st.number_input("Public Records",         0,  10,   0)
        pub_rec_bk     = st.number_input("Bankruptcies",           0,   5,   0)
        inq_last_6mths = st.number_input("Inquiries (6mo)",        0,  10,   0)
        mort_acc       = st.number_input("Mortgage Accounts",      0,  20,   0)

    st.divider()

    if st.button("🔍 Evaluate Risk", type="primary", use_container_width=True):

        fields = {
            'loan_amnt': loan_amnt, 'int_rate': int_rate, 'term': term,
            'purpose': purpose, 'grade': grade, 'sub_grade': sub_grade,
            'annual_inc': annual_inc, 'emp_length': emp_length,
            'home_ownership': home_own, 'verification_status': verification,
            'application_type': application_type, 'addr_state': addr_state,
            'fico_low': fico_low, 'fico_high': fico_high,
            'dti': dti, 'revol_util': revol_util, 'revol_bal': revol_bal,
            'open_acc': open_acc, 'total_acc': total_acc,
            'delinq_2yrs': delinq_2yrs, 'pub_rec': pub_rec,
            'pub_rec_bankruptcies': pub_rec_bk,
            'inq_last_6mths': inq_last_6mths, 'mort_acc': mort_acc,
            'credit_history_months': 120,
        }

        try:
            input_df = build_input(fields)

            # Predict directly with raw booster — no sklearn wrapper
            prob = booster.predict(input_df)[0]
            score, grade_label, decision = get_risk_details(prob)

            # Results
            st.subheader("📊 Risk Assessment Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Risk Score",          score)
            r2.metric("Default Probability", f"{prob:.1%}")
            r3.metric("Risk Grade",          grade_label.split('—')[0].strip())
            r4.metric("Decision",            decision)

            if "APPROVED" in decision:
                st.success(f"### {decision}  |  {grade_label}")
            elif "REVIEW" in decision:
                st.warning(f"### {decision}  |  {grade_label}")
            else:
                st.error(f"### {decision}  |  {grade_label}")

            # Risk gauge
            pct       = (score - 300) / 550
            bar_color = "#2ecc71" if pct > 0.6 else "#f39c12" if pct > 0.35 else "#e74c3c"
            st.markdown(f"""
            <div style="background:#eee;border-radius:10px;height:30px;width:100%">
              <div style="background:{bar_color};width:{pct*100:.1f}%;height:30px;
                          border-radius:10px;text-align:center;line-height:30px;
                          color:white;font-weight:bold">{score}</div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:4px">
              <span>300 — High Risk</span><span>575</span><span>850 — Low Risk</span>
            </div>
            """, unsafe_allow_html=True)

            # Key factors
            st.markdown("#### 🔑 Key Risk Factors")
            fc1, fc2, fc3 = st.columns(3)
            fc1.markdown(f"**DTI:** {'✅ Good' if dti < 20 else '⚠️ Elevated' if dti < 35 else '🔴 High'}")
            fc2.markdown(f"**FICO:** {'✅ Good' if fico_low >= 700 else '⚠️ Fair' if fico_low >= 650 else '🔴 Poor'}")
            fc3.markdown(f"**Int Rate:** {'✅ Good' if int_rate < 12 else '⚠️ Elevated' if int_rate < 20 else '🔴 High'}")
            fc1.markdown(f"**Delinquencies:** {'✅ None' if delinq_2yrs == 0 else '🔴 Present'}")
            fc2.markdown(f"**Public Records:** {'✅ None' if pub_rec == 0 else '🔴 Present'}")
            fc3.markdown(f"**Revol. Util:** {'✅ Good' if revol_util < 30 else '⚠️ Elevated' if revol_util < 60 else '🔴 High'}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

with tab2:
    st.markdown("""
    ### About This System
    - **Model:** LightGBM trained on 1.3M+ Lending Club loans (2007–2018)
    - **AUC:** 0.9587 on held-out test set
    - **Features:** 114 features including credit history, income ratios, FICO, delinquency data
    - **Risk Score:** 300–850 scale (higher = lower risk, similar to FICO)

    ### Decision Thresholds
    | Score | Grade | Decision |
    |-------|-------|----------|
    | 750–850 | A — Excellent | ✅ Approve |
    | 700–749 | B — Good | ✅ Approve |
    | 650–699 | C — Fair | ✅ Approve |
    | 600–649 | D — Poor | ⚠️ Manual Review |
    | 550–599 | E — Very Poor | ⚠️ Manual Review |
    | 300–549 | F — High Risk | ❌ Reject |
    """)