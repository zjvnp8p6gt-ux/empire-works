
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================
# 台股分析網站 V1
# =========================

st.set_page_config(
    page_title="TW Market Intelligence V1",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color: #10151f; color: #e8edf5;}
[data-testid="stMetric"] {
    background-color: #1b2533;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 TW Market Intelligence V1")
st.caption("台灣股票市場分析系統｜歷史資料研究工具")

STOCKS = {
    "2330": ("台積電", "2330.TW"),
    "2454": ("聯發科", "2454.TW"),
    "2308": ("台達電", "2308.TW"),
    "2317": ("鴻海", "2317.TW"),
    "8261": ("富鼎", "8261.TWO")
}

# =========================
# 技術指標計算
# =========================

def add_indicators(df):
    df = df.copy()
    df = df.sort_values("日期")

    close = df["收盤價"].astype(float)

    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI14"] = 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    df["日報酬率(%)"] = close.pct_change() * 100
    df["20日漲跌幅(%)"] = close.pct_change(20) * 100

    return df


# =========================
# 資料取得：Yahoo Finance
# =========================

@st.cache_data(ttl=3600)
def get_market_data():
    all_data = []

    for code, (name, ticker) in STOCKS.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y", auto_adjust=False)

            if hist.empty:
                continue

            temp = hist.reset_index()

            temp = temp.rename(columns={
                "Date": "日期",
                "Datetime": "日期",
                "Close": "收盤價",
                "Volume": "成交股數",
                "Open": "開盤價",
                "High": "最高價",
                "Low": "最低價"
            })

            temp["日期"] = pd.to_datetime(
                temp["日期"]
            ).dt.tz_localize(None)

            temp["股票代號"] = code
            temp["股票名稱"] = name

            all_data.append(temp[
                ["股票代號", "股票名稱", "日期",
                 "開盤價", "最高價", "最低價",
                 "收盤價", "成交股數"]
            ])

        except Exception:
            continue

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


# =========================
# CSV 資料整理
# =========================

def prepare_csv(file):
    df = pd.read_csv(file)

    required = ["股票代號", "股票名稱", "日期", "收盤價"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV 缺少必要欄位：" + "、".join(missing)
        )

    df["股票代號"] = (
        df["股票代號"].astype(str)
        .str.replace(".0", "", regex=False)
    )

    df["日期"] = pd.to_datetime(
        df["日期"], errors="coerce"
    )

    df["收盤價"] = pd.to_numeric(
        df["收盤價"], errors="coerce"
    )

    df = df.dropna(subset=["日期", "收盤價"])

    return df


# =========================
# 側邊欄：資料來源
# =========================

st.sidebar.header("⚙️ 分析設定")

source = st.sidebar.radio(
    "資料來源",
    ["自動更新歷史行情", "上傳 CSV 歷史資料"]
)

if st.sidebar.button("🔄 重新取得資料"):
    st.cache_data.clear()

df_all = pd.DataFrame()

if source == "自動更新歷史行情":
    with st.spinner("正在取得五檔股票歷史資料..."):
        df_all = get_market_data()

    if df_all.empty:
        st.error("目前無法取得行情資料，請稍後重試或改用 CSV。")
        st.stop()

else:
    uploaded = st.sidebar.file_uploader(
        "上傳歷史資料 CSV",
        type=["csv"]
    )

    if uploaded is None:
        st.info("請上傳 CSV。必要欄位：股票代號、股票名稱、日期、收盤價。")
        st.stop()

    try:
        df_all = prepare_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()


# =========================
# 計算全部股票指標
# =========================

result = []

for code, group in df_all.groupby("股票代號"):
    group = group.copy()
    group = add_indicators(group)
    result.append(group)

if not result:
    st.error("沒有可供分析的資料。")
    st.stop()

df_all = pd.concat(result, ignore_index=True)

# =========================
# 建立個股總覽
# =========================

summary = []

for code, group in df_all.groupby("股票代號"):
    group = group.sort_values("日期")
    last = group.iloc[-1]

    first_price = group.iloc[0]["收盤價"]
    latest_price = last["收盤價"]

    total_return = (
        (latest_price / first_price) - 1
    ) * 100 if first_price else np.nan

    daily_returns = group["收盤價"].pct_change().dropna()

    mean_daily = (
        daily_returns.mean() * 100
        if len(daily_returns) else np.nan
    )

    volatility = (
        daily_returns.std() * 100
        if len(daily_returns) > 1 else np.nan
    )

    annual_volatility = (
        volatility * np.sqrt(252)
        if pd.notna(volatility) else np.nan
    )

    # MA 結構：最高 4 分
    ma_score = 0

    if pd.notna(last["MA20"]):
        if (
            last["收盤價"] > last["MA5"]
            and last["MA5"] > last["MA10"]
            and last["MA10"] > last["MA20"]
        ):
            ma_score = 4
        elif last["收盤價"] > last["MA20"]:
            ma_score = 2
        elif last["收盤價"] > last["MA10"]:
            ma_score = 1

    # RSI：最高 2 分
    rsi = last["RSI14"]
    rsi_score = 0

    if pd.notna(rsi):
        if 50 <= rsi <= 70:
            rsi_score = 2
        elif 40 <= rsi < 50:
            rsi_score = 1

    # MACD：最高 2 分
    macd_score = 0

    if (
        pd.notna(last["MACD"])
        and pd.notna(last["Signal"])
        and last["MACD"] > last["Signal"]
    ):
        macd_score = 2

    # 20 日動能：最高 2 分
    momentum = last["20日漲跌幅(%)"]
    momentum_score = 0

    if pd.notna(momentum):
        if momentum > 0:
            momentum_score = 2
        elif momentum >= -3:
            momentum_score = 1

    total_score = (
        ma_score + rsi_score
        + macd_score + momentum_score
    )

    summary.append({
        "股票代號": code,
        "股票名稱": group.iloc[-1]["股票名稱"],
        "日期": last["日期"],
        "收盤價": latest_price,
        "MA5": last["MA5"],
        "MA10": last["MA10"],
        "MA20": last["MA20"],
        "RSI14": rsi,
        "MACD": last["MACD"],
        "Signal": last["Signal"],
        "MACD_Hist": last["MACD_Hist"],
        "20日漲跌幅(%)": momentum,
        "累積漲跌幅(%)": total_return,
        "平均日報酬率(%)": mean_daily,
        "日波動率(%)": volatility,
        "年化波動率(%)": annual_volatility,
        "均線分數": ma_score,
        "RSI分數": rsi_score,
        "MACD分數": macd_score,
        "20日動能分數": momentum_score,
        "10分制綜合分數": total_score
    })

summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "10分制綜合分數",
    ascending=False
).reset_index(drop=True)

summary_df.insert(
    0, "研究排名", range(1, len(summary_df) + 1)
)

# =========================
# 主要頁面
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 綜合總覽",
    "📈 個股走勢",
    "🧮 技術指標",
    "⚖️ 報酬與風險",
    "📝 自動研究報告"
])

# =========================
# TAB 1 綜合總覽
# =========================

with tab1:
    st.subheader("五檔股票最新行情總覽")

    c1, c2, c3 = st.columns(3)

    c1.metric("分析股票數量", len(summary_df))
    c2.metric("歷史資料筆數", len(df_all))
    c3.metric(
        "平均技術評分",
        f"{summary_df['10分制綜合分數'].mean():.2f} / 10"
    )

    st.dataframe(
        summary_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("10 分制綜合技術評分排名")

    score_chart = px.bar(
        summary_df,
        x="股票名稱",
        y="10分制綜合分數",
        color="10分制綜合分數",
        text="10分制綜合分數",
        title="五檔股票技術評分"
    )

    score_chart.update_yaxes(range=[0, 10])
    st.plotly_chart(score_chart, use_container_width=True)

    st.subheader("研究排名")

    st.dataframe(
        summary_df[[
            "研究排名", "股票代號", "股票名稱",
            "20日漲跌幅(%)", "10分制綜合分數"
        ]].round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================
# TAB 2 個股走勢
# =========================

with tab2:
    st.subheader("個股歷史走勢分析")

    selected = st.selectbox(
        "選擇股票",
        summary_df["股票代號"].tolist(),
        format_func=lambda x:
            f"{x} {STOCKS.get(x, (x,))[0]}"
    )

    stock_df = df_all[
        df_all["股票代號"] == selected
    ].sort_values("日期")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock_df["日期"],
        y=stock_df["收盤價"],
        name="收盤價"
    ))

    for col in ["MA5", "MA10", "MA20"]:
        fig.add_trace(go.Scatter(
            x=stock_df["日期"],
            y=stock_df[col],
            name=col
        ))

    fig.update_layout(
        title=f"{selected} 歷史收盤價與均線",
        xaxis_title="日期",
        yaxis_title="價格",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("歷史資料")

    st.dataframe(
        stock_df.tail(30).round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================
# TAB 3 技術指標
# =========================

with tab3:
    st.subheader("RSI 與 MACD 分析")

    selected2 = st.selectbox(
        "選擇分析股票",
        summary_df["股票代號"].tolist(),
        key="indicator_stock"
    )

    indicator_df = df_all[
        df_all["股票代號"] == selected2
    ].sort_values("日期")

    fig_rsi = px.line(
        indicator_df,
        x="日期",
        y="RSI14",
        title="RSI14"
    )

    fig_rsi.add_hline(y=70, line_dash="dash")
    fig_rsi.add_hline(y=30, line_dash="dash")

    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()

    fig_macd.add_trace(go.Scatter(
        x=indicator_df["日期"],
        y=indicator_df["MACD"],
        name="MACD"
    ))

    fig_macd.add_trace(go.Scatter(
        x=indicator_df["日期"],
        y=indicator_df["Signal"],
        name="Signal"
    ))

    fig_macd.add_trace(go.Bar(
        x=indicator_df["日期"],
        y=indicator_df["MACD_Hist"],
        name="MACD Histogram"
    ))

    fig_macd.update_layout(
        title="MACD 技術指標",
        template="plotly_dark"
    )

    st.plotly_chart(fig_macd, use_container_width=True)


# =========================
# TAB 4 報酬與風險
# =========================

with tab4:
    st.subheader("歷史報酬與風險比較")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.bar(
                summary_df,
                x="股票名稱",
                y="累積漲跌幅(%)",
                title="歷史累積報酬率"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.bar(
                summary_df,
                x="股票名稱",
                y="年化波動率(%)",
                title="年化波動率"
            ),
            use_container_width=True
        )

    st.subheader("報酬與風險數據")

    st.dataframe(
        summary_df[[
            "股票代號", "股票名稱",
            "累積漲跌幅(%)",
            "平均日報酬率(%)",
            "日波動率(%)",
            "年化波動率(%)"
        ]].round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================
# TAB 5 自動研究報告
# =========================

with tab5:
    st.subheader("自動研究報告")

    st.write(
        f"資料涵蓋日期："
        f"{df_all['日期'].min().date()} 至 "
        f"{df_all['日期'].max().date()}"
    )

    st.write(f"歷史資料總筆數：{len(df_all)}")

    best = summary_df.iloc[0]

    st.markdown("### 技術評分排名第一")

    st.write(
        f"{best['股票代號']} {best['股票名稱']}，"
        f"技術評分 {best['10分制綜合分數']:.0f} 分。"
    )

    st.markdown("### 五檔股票研究摘要")

    for _, row in summary_df.iterrows():
        st.markdown(
            f"""
            **{row['研究排名']}. {row['股票代號']} {row['股票名稱']}**

            - 收盤價：{row['收盤價']:.2f}
            - 20日漲跌幅：{row['20日漲跌幅(%)']:.2f}%
            - RSI14：{row['RSI14']:.2f}
            - MACD：{row['MACD']:.2f}
            - 綜合評分：{row['10分制綜合分數']:.0f}/10
            """
        )

    st.markdown("### 評分規則")

    st.write("""
    - 均線結構：最高 4 分
    - RSI 指標：最高 2 分
    - MACD 指標：最高 2 分
    - 20 日動能：最高 2 分
    - 總分：10 分
    """)

    report = summary_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 匯出綜合研究報告 CSV",
        data=report,
        file_name="TW_Market_Research_Report.csv",
        mime="text/csv"
    )

    history_csv = df_all.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 匯出完整歷史分析資料",
        data=history_csv,
        file_name="TW_Market_History.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "TW Market Intelligence V1｜"
    "歷史資料分析工具，非即時報價或投資建議。"
)
