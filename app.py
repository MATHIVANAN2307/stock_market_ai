import streamlit as st
import plotly.graph_objects as go
import torch

from data_utils import load_data, add_moving_averages, support_resistance, prepare_data
from signals import generate_signals
from model import BiLSTMAttention

st.set_page_config(layout="wide")
st.title("📊 AI Market Trend Analysis Dashboard")

ticker = st.text_input("Enter Stock Symbol", "GOOG")

@st.cache_data
def get_data(ticker):
    df = load_data(ticker)
    df = add_moving_averages(df)
    return df

df = get_data(ticker)
support, resistance = support_resistance(df)
buy, sell = generate_signals(df)

# ==========================
# 1️⃣ CANDLESTICK DASHBOARD
# ==========================
st.header("📌 Candlestick Price Analysis")

fig1 = go.Figure()
fig1.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
))
st.plotly_chart(fig1, use_container_width=True)

# ==========================
# 2️⃣ MOVING AVERAGES
# ==========================
st.header("📈 Moving Averages (20 / 50 / 200)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig2.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
fig2.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
fig2.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200"))
st.plotly_chart(fig2, use_container_width=True)

# ==========================
# 3️⃣ SUPPORT & RESISTANCE
# ==========================
st.header("🧱 Support & Resistance Levels")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig3.add_trace(go.Scatter(x=df.index, y=support, name="Support", line=dict(dash="dot")))
fig3.add_trace(go.Scatter(x=df.index, y=resistance, name="Resistance", line=dict(dash="dot")))
st.plotly_chart(fig3, use_container_width=True)

# ==========================
# 4️⃣ BUY / SELL SIGNALS
# ==========================
st.header("💰 Buy / Sell Signal Analysis")

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig4.add_trace(go.Scatter(
    x=df.index, y=buy,
    mode="markers", name="Buy",
    marker=dict(color="green", size=10, symbol="triangle-up")
))
fig4.add_trace(go.Scatter(
    x=df.index, y=sell,
    mode="markers", name="Sell",
    marker=dict(color="red", size=10, symbol="triangle-down")
))
st.plotly_chart(fig4, use_container_width=True)

# ==========================
# 5️⃣ AI PREDICTION (LSTM)
# ==========================
st.header("🤖 AI Price Prediction (BiLSTM + Attention)")

X, _, scaler = prepare_data(df)
model = BiLSTMAttention()
model.load_state_dict(
    torch.load("bilstm_model.pth", map_location=torch.device("cpu"))
)
model.eval()

last_seq = torch.tensor(X[-1:], dtype=torch.float32)
pred = model(last_seq).detach().numpy()
pred_price = scaler.inverse_transform(pred)[0][0]

current_price = df["Close"].iloc[-1]

st.write(f"Current Price: **{current_price:.2f}**")
st.write(f"Predicted Price: **{pred_price:.2f}**")
