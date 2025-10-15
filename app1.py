import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import requests


# ======================
# 📌 Streamlit 設定
# ======================
st.set_page_config(page_title="Amazon Sales Forecasting", layout="wide")
st.title("📊 Amazon Sales Forecasting Dashboard")
st.markdown("This is a **business decision-driven** sales forecasting dashboard. You can upload Amazon sales data, choose a model (BiLSTM / TCN / XGBoost), and get future trend insights.")

# ======================
# 📌 載入模型
# ======================
@st.cache_resource
def load_tcn_model():
    return load_model("best_tcn_model.keras", compile=False)

@st.cache_resource
def load_bilstm_model():
    return load_model("best_bilstm_model.keras", compile=False)

@st.cache_resource
def load_xgb_model():
    return joblib.load("best_xgb_model.pkl")

tcn_model = load_tcn_model()
bilstm_model = load_bilstm_model()
xgb_model = load_xgb_model()

# ======================
# 📌 工具函數
# ======================
def preprocess_time_series(df, model_type, window_size, output_days):
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])
    daily_sales = df.groupby("Invoice Date")["Sales Amount"].sum().reset_index()
    daily_sales.set_index("Invoice Date", inplace=True)

    # === BiLSTM 用 7日均線, TCN 用原始值 ===
    if model_type == "BiLSTM":
        daily_sales["Sales_MA7"] = daily_sales["Sales Amount"].rolling(window=7, min_periods=1).mean()
        target_series = daily_sales[["Sales_MA7"]]
    else:
        target_series = daily_sales[["Sales Amount"]]

    # === Split raw for scaling (70/10/20) ===
    n = len(target_series)
    split1 = int(0.7 * n)
    split2 = int(0.8 * n)
    train_raw = target_series.values[:split1]
    val_raw   = target_series.values[split1:split2]
    test_raw  = target_series.values[split2:]

    # === Scaler (fit only on train) ===
    scaler = MinMaxScaler()
    scaler.fit(train_raw)
    scaled = scaler.transform(target_series.values)

    # === Sequence function ===
    def create_sequences(data, window_size, output_size):
        x, y = [], []
        for i in range(len(data) - window_size - output_size + 1):
            x.append(data[i:i+window_size])
            y.append(data[i+window_size:i+window_size+output_size])
        return np.array(x), np.array(y)

    X_all, y_all = create_sequences(scaled, window_size, output_days)

    # === Split scaled sequences ===
    split1 = int(0.7 * len(X_all))
    split2 = int(0.8 * len(X_all))
    X_train, y_train = X_all[:split1], y_all[:split1]
    X_val, y_val     = X_all[split1:split2], y_all[split1:split2]
    X_test, y_test   = X_all[split2:], y_all[split2:]

    # Reshape for LSTM/TCN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val   = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, daily_sales

def add_xgb_features(df):
    """XGBoost 特徵工程：日期 + lag + rolling"""
    df["Day"] = df.index.day
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    df["DayOfWeek"] = df.index.dayofweek
    df["IsWeekend"] = (df.index.dayofweek >= 5).astype(int)
    df["IsMonthStart"] = df.index.is_month_start.astype(int)
    df["IsMonthEnd"] = df.index.is_month_end.astype(int)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"lag{lag}"] = df["Sales Amount"].shift(lag)

    # Rolling features
    df["roll7_mean"] = df["Sales Amount"].shift(1).rolling(window=7).mean()
    df["roll30_mean"] = df["Sales Amount"].shift(1).rolling(window=30).mean()
    df["roll7_std"] = df["Sales Amount"].shift(1).rolling(window=7).std()
    df["roll30_std"] = df["Sales Amount"].shift(1).rolling(window=30).std()

    df = df.dropna()
    return df

def ai_assistant_recommendation(next_forecast, user_question=None):
    """Call local Ollama (Gemma3:4b) for recommendations"""
    try:
        question_part = f"\nUser Question: {user_question}" if user_question else ""
        prompt = f"""
        You are a business strategy AI. Here are the next 7 days forecasted sales:

        {next_forecast.to_string(index=False)}

        {question_part}

        Based on this, recommend inventory planning, promotions, and operational strategy.
        Keep it concise in bullet points.
        """
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
            timeout=120
        )
        try:
            data = response.json()
            return data.get("response", "⚠️ Ollama 沒有回覆內容")
        except json.JSONDecodeError:
            return f"⚠️ Failed to parse Ollama response: {response.text}"
    except Exception as e:
        return f"⚠️ Cannot connect to Ollama: {e}"


# ======================
# 📌 File Upload & Model Selection
# ======================
uploaded_file = st.file_uploader("📂 Upload Amazon Sales Data (CSV)", type=["csv"])
model_option = st.selectbox("Select Model", ["TCN", "BiLSTM", "XGBoost"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully")
    st.write("First 5 rows of data:：")
    st.dataframe(df.head())

    if "Invoice Date" not in df.columns or "Sales Amount" not in df.columns:
        st.error("❌ Dataset must include 'Invoice Date' and 'Sales Amount' columns")
    else:
        # ======================
        # 📌 BiLSTM / TCN
        # ======================
        if model_option in ["TCN", "BiLSTM"]:
            model = tcn_model if model_option == "TCN" else bilstm_model

            # Model input shape
            model_input_shape = model.input_shape
            WINDOW_SIZE = model_input_shape[1]
            OUTPUT_DAYS = model.output_shape[1]

            # === 前處理 ===
            X_train, y_train, X_val, y_val, X_test, y_test, scaler, daily_sales = preprocess_time_series(
                df, model_option, WINDOW_SIZE, OUTPUT_DAYS
            )

            # === Prediction on test set ===
            y_pred = model.predict(X_test)
            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
            y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

            # === 評估 (Day+1) ===
            mae = mean_absolute_error(y_true_inv[:, 0], y_pred_inv[:, 0])
            rmse = mean_squared_error(y_true_inv[:, 0], y_pred_inv[:, 0], squared=False)
            r2 = r2_score(y_true_inv[:, 0], y_pred_inv[:, 0])

            # === 未來 7 天預測 ===
            if model_option == "BiLSTM":
                last_window = scaler.transform(daily_sales[["Sales_MA7"]].values[-WINDOW_SIZE:])
            else:  # TCN
                last_window = scaler.transform(daily_sales[["Sales Amount"]].values[-WINDOW_SIZE:])

            last_window = last_window.reshape((1, WINDOW_SIZE, 1))
            future_pred = model.predict(last_window)
            future_pred_inv = scaler.inverse_transform(future_pred.reshape(-1,1)).reshape(future_pred.shape)

            forecast_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=OUTPUT_DAYS)
            next_forecast = pd.DataFrame({
                "Date": forecast_dates,
                "Predicted Sales Amount": future_pred_inv.flatten()
            })

        # ======================
        # 📌 XGBoost
        # ======================
        elif model_option == "XGBoost":
            daily_sales = df.groupby("Invoice Date")["Sales Amount"].sum().reset_index()
            daily_sales["Invoice Date"] = pd.to_datetime(daily_sales["Invoice Date"])
            daily_sales.set_index("Invoice Date", inplace=True)

            # 加特徵工程
            daily_sales_features = add_xgb_features(daily_sales.copy())

            # Train/Test split
            train_size = int(len(daily_sales_features) * 0.8)
            train = daily_sales_features.iloc[:train_size]
            test = daily_sales_features.iloc[train_size:]

            X_train = train.drop(columns=["Sales Amount"])
            y_train = train["Sales Amount"]
            X_test = test.drop(columns=["Sales Amount"])
            y_test = test["Sales Amount"]

            # Prediction (回測)
            y_pred = xgb_model.predict(X_test)

            # 評估
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            # === 未來 7 天預測 (遞推方式) ===
            last_date = daily_sales.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)

            future_df = daily_sales.copy()
            future_preds = []

            for date in future_dates:
                new_row = {"Sales Amount": np.nan}
                tmp_df = pd.concat([future_df, pd.DataFrame([new_row], index=[date])])

                tmp_df_feat = add_xgb_features(tmp_df.copy())
                X_future = tmp_df_feat.drop(columns=["Sales Amount"]).iloc[[-1]]

                pred = xgb_model.predict(X_future)[0]
                future_preds.append(pred)

                future_df.loc[date, "Sales Amount"] = pred

            next_forecast = pd.DataFrame({
                "Date": future_dates,
                "Predicted Sales Amount": future_preds
            })

        # ======================
        # 📌 模型性能指標
        # ======================
        st.subheader("📌 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:,.2f}")
        col2.metric("RMSE", f"{rmse:,.2f}")
        col3.metric("R²", f"{r2:.4f}")

        # ======================
        # 📌 未來 7 日預測
        # ======================
        st.subheader("📊 Next 7 Days Forecast")
        st.dataframe(next_forecast)

        # KPI
        latest_val = next_forecast["Predicted Sales Amount"].iloc[-1]
        max_val = next_forecast["Predicted Sales Amount"].max()
        avg_val = next_forecast["Predicted Sales Amount"].mean()
        max_date = next_forecast.loc[next_forecast["Predicted Sales Amount"].idxmax(), "Date"]

        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Latest Forecast", f"{latest_val:,.0f}")
        col2.metric("🔥 Peak Sales Day", f"{max_date.strftime('%Y-%m-%d')}", f"{max_val:,.0f}")
        col3.metric("📊 7-Day Average", f"{avg_val:,.0f}")

        # ======================
        # 📌 視覺化
        # ======================
        st.subheader("📈 Forecast Visualization")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(next_forecast["Date"], next_forecast["Predicted Sales Amount"], color="skyblue")
        ax.set_title("Next 7 Days Predicted Sales")
        ax.set_ylabel("Sales Amount")
        st.pyplot(fig)

        # ======================
        # 📌 商業洞察
        # ======================
        trend = "上升" if next_forecast["Predicted Sales Amount"].iloc[-1] > next_forecast["Predicted Sales Amount"].iloc[0] else "下降"
        st.subheader("💡 Business Insight")
        st.write(f"Sales over the next 7 days show a **{trend} trend**，with the peak expected on **{max_date.strftime('%Y-%m-%d')}** (約 {max_val:,.0f})。")
        st.write("👉 Recommendation: Plan inventory and logistics in advance, and strengthen promotions before peak days.")

        # ======================
        # 📌 Download Results
        # ======================
        csv = next_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 7-Day Forecast CSV",
            data=csv,
            file_name=f"{model_option.lower()}_7days_forecast.csv",
            mime="text/csv",
        )

        # ======================
        # 📌 AI Assistant (每個模型自動分析)
        # ======================
        st.subheader("🤖 AI Assistant Recommendation")
        default_suggestion = ai_assistant_recommendation(next_forecast)
        
        with st.container():
             st.markdown(
                 f"""
        <div style='padding:15px; border-radius:10px; background-color:#f0f2f6;'>
            <h4 style='color:#2c3e50;'>📊 AI Analysis</h4>
            <p style='font-size:15px; color:#34495e;'>{default_suggestion}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
        # Step 2: 允許用戶輸入問題
        st.markdown("### 💬 Ask Your Question")
        user_question = st.text_area("Enter your own question (e.g. How to reduce stock overages?)", "")
        if user_question:
             user_suggestion = ai_assistant_recommendation(next_forecast, user_question)
             st.markdown("### 🎯AI Answer to Your Question")
             st.success(user_suggestion)



            

        
        
