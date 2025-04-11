import streamlit as st
import numpy as np
from scipy.stats import nbinom, poisson

# 常數設定
NPB_STD_DEV = 2.3
NPB_NB_R = 9.0
KBO_STD_DEV = 2.8
KBO_NB_R = 7.5
NUM_SIMULATIONS = 100000

st.set_page_config(page_title="NPB/KBO 模擬分析", layout="centered")
st.title("🏟️ NPB/KBO 總分模擬分析系統")

st.markdown("本工具透過三種統計模型預測比賽總分是否大於盤口，適用於日本職棒 (NPB) 與韓國職棒 (KBO)。")

league = st.selectbox("選擇聯盟", ["NPB（日職）", "KBO（韓職）"])
league_key = "NPB" if "NPB" in league else "KBO"

time_period = st.selectbox("比賽時段", ["日間", "晚間"])

st.subheader("主隊資料")
home = {}
home["name"] = st.text_input("主隊名稱", "主隊")
home["time_avg"] = st.number_input("近期場均得分", min_value=0.0)
home["base_avg"] = st.number_input("整體場均得分", min_value=0.0)
home["allow"] = st.number_input("主隊場均失分", min_value=0.0)
home["over_rate"] = st.slider("大分過盤率 (%)", 0, 100) / 100
home["team_batting"] = st.number_input("團隊打擊率", min_value=0.000, max_value=1.000, value=0.270)
home["team_obp"] = st.number_input("團隊上壘率", min_value=0.000, max_value=1.000, value=0.340)
home["pitcher"] = {
    "era": st.number_input("先發投手防禦率", min_value=0.0),
    "baa": st.number_input("先發投手被打擊率", min_value=0.000, max_value=1.000, value=0.250)
}

st.subheader("客隊資料")
away = {}
away["name"] = st.text_input("客隊名稱", "客隊")
away["time_avg"] = st.number_input("近期場均得分（客隊）", min_value=0.0)
away["base_avg"] = st.number_input("整體場均得分（客隊）", min_value=0.0)
away["allow"] = st.number_input("客隊場均失分", min_value=0.0)
away["over_rate"] = st.slider("大分過盤率（客隊） (%)", 0, 100, key="客隊") / 100
away["team_batting"] = st.number_input("團隊打擊率（客隊）", min_value=0.000, max_value=1.000, value=0.270)
away["team_obp"] = st.number_input("團隊上壘率（客隊）", min_value=0.000, max_value=1.000, value=0.340)
away["pitcher"] = {
    "era": st.number_input("先發投手防禦率（客隊）", min_value=0.0),
    "baa": st.number_input("先發投手被打擊率（客隊）", min_value=0.000, max_value=1.000, value=0.250)
}

target = st.number_input("盤口總分", min_value=0.0)

def weighted_score(data, opp_pitcher, league_factor):
    era_impact = (4.5 / max(0.01, opp_pitcher['era'])) * 0.7
    baa_impact = (0.3 / max(0.001, opp_pitcher['baa'])) * 0.3
    pitcher_factor = min(2.0, era_impact + baa_impact)

    batting_impact = (data['team_batting'] / max(0.001, 0.270)) * 0.5
    obp_impact = (data['team_obp'] / max(0.001, 0.340)) * 0.5

    base_score = (data['time_avg'] * 0.5 + data['base_avg'] * 0.3 + data['allow'] * 0.2)

    return max(0.1, base_score * league_factor *
               (0.8 + 0.2 * data['over_rate']) *
               (1.0 / max(0.1, pitcher_factor)) *
               (0.6 + batting_impact + obp_impact))

def run_models(hs, as_, target, league_key):
    std_dev = NPB_STD_DEV if league_key == 'NPB' else KBO_STD_DEV
    nb_r = NPB_NB_R if league_key == 'NPB' else KBO_NB_R

    home_std = std_dev * (hs / max(0.1, hs + as_))
    away_std = std_dev * (as_ / max(0.1, hs + as_))
    mc = np.mean(np.random.normal(hs, home_std, NUM_SIMULATIONS) + np.random.normal(as_, away_std, NUM_SIMULATIONS) > target) * 100

    def gen_nb(mean): 
        p = nb_r / max(0.1, nb_r + mean)
        return nbinom.rvs(nb_r, p, size=NUM_SIMULATIONS)
    
    nb = np.mean(gen_nb(hs) + gen_nb(as_) > target) * 100

    poisson = np.mean(np.random.poisson(hs, NUM_SIMULATIONS) + np.random.poisson(as_, NUM_SIMULATIONS) > target) * 100

    if league_key == 'NPB':
        final = mc * 0.5 + nb * 0.35 + poisson * 0.15
    else:
        final = mc * 0.6 + nb * 0.25 + poisson * 0.15

    return mc, nb, poisson, final

if st.button("🔍 開始分析"):
    league_factor = 0.95 if league_key == 'NPB' else 1.05
    hs = weighted_score(home, away['pitcher'], league_factor)
    as_ = weighted_score(away, home['pitcher'], league_factor)

    mc, nb, poi, final = run_models(hs, as_, target, league_key)

    st.subheader("📊 模型分析結果")
    st.markdown(f"**預期得分**：主隊 {hs:.2f} vs 客隊 {as_:.2f}")
    st.markdown(f"**預期總分**：{hs + as_:.2f}（盤口 {target}）")
    st.markdown(f"- 蒙地卡羅模擬：`{mc:.1f}%`")
    st.markdown(f"- 負二項分布模擬：`{nb:.1f}%`")
    st.markdown(f"- 泊松分布模擬：`{poi:.1f}%`")
    st.markdown(f"🎯 **綜合推薦機率：`{final:.1f}%`**")

    if final >= 55:
        st.success("🔥 建議：偏向 **大分**")
    elif final <= 45:
        st.warning("❄️ 建議：偏向 **小分**")
    else:
        st.info("⚖️ 建議：屬於五五波區間，請參考其他因素")
