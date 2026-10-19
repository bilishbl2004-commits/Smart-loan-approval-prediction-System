import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Smart Loan Approval AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {color: #1f77b4; font-size: 3rem; font-weight: bold; text-align: center;}
    .metric-approved {background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .metric-rejected {background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .advice-box {background: linear-gradient(135deg, #fff3cd, #ffeaa7); padding: 1.5rem; border-radius: 15px; border-left: 6px solid #ffc107; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .stMetric > label {font-size: 1.2rem; font-weight: bold;}
    .stMetric > div > div {font-size: 2.5rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Database with AUTO-FIX
@st.cache_resource
def init_db():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    c = conn.cursor()
    
    # Create users table
    c.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)")
    
    # Check and fix history table
    c.execute("PRAGMA table_info(history)")
    cols = [col[1] for col in c.fetchall()]
    
    if 'probability' not in cols or 'advice' not in cols or 'timestamp' not in cols:
        print("🔧 Fixing database schema...")
        c.execute("DROP TABLE IF EXISTS history")
        c.execute("""
            CREATE TABLE history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                prediction TEXT,
                probability REAL,
                advice TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print("✅ Database fixed!")
    
    conn.commit()
    return conn, c

conn, c = init_db()

def hash_password(password): 
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
             (username, hash_password(password)))
    return c.fetchone()

def add_history(username, prediction, prob, advice):
    c.execute("""
        INSERT INTO history (username, prediction, probability, advice) 
        VALUES (?, ?, ?, ?)
    """, (username, prediction, prob, advice))
    conn.commit()

def view_history(username):
    c.execute("""
        SELECT prediction, probability, advice, timestamp 
        FROM history WHERE username=? 
        ORDER BY timestamp DESC LIMIT 10
    """, (username,))
    return c.fetchall()

# Load models
@st.cache_resource
def load_models():
    model = pickle.load(open("model_files/loan_model.pkl", "rb"))
    encoders = pickle.load(open("model_files/encoders.pkl", "rb"))
    columns = pickle.load(open("model_files/columns.pkl", "rb"))
    scaler = pickle.load(open("model_files/scaler.pkl", "rb"))
    importance = pd.read_csv("model_files/importance.csv")
    return model, encoders, columns, scaler, importance

model, encoders, columns, scaler, importance = load_models()

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# =============== LOGIN/REGISTER ===============
if not st.session_state.logged_in:
    st.markdown('<h1 class="main-header">🏦 Smart Loan Approval AI</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 **Register**")
        new_user = st.text_input("👉 Username", key="reg_user")
        new_pass = st.text_input("🔒 Password", type="password", key="reg_pass")
        if st.button("✅ Create Account", use_container_width=True, type="primary"):
            if new_user and new_pass:
                if add_user(new_user, new_pass):
                    st.success("🎉 Account created successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Username already exists!")
            else:
                st.warning("⚠️ Please fill all fields")
    
    with col2:
        st.subheader("🔐 **Login**")
        username = st.text_input("👉 Username", key="login_user")
        password = st.text_input("🔒 Password", type="password", key="login_pass")
        if st.button("🚀 Login", use_container_width=True):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("🎉 Welcome back!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials!")

# =============== MAIN DASHBOARD ===============
else:
    # Header
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; border-radius: 20px; margin-bottom: 2rem;'>
        <h1>👋 Welcome, {st.session_state.username}!</h1>
        <p>AI-Powered Loan Approval Predictor with Smart Advice</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    menu = st.sidebar.selectbox("🌟 Choose Action", 
                               ["🔮 Get Loan Prediction", "📊 View History", "⚡ Key Factors", "🚪 Logout"])
    
    if menu == "🔮 Get Loan Prediction":
        st.markdown("## 🎯 Enter Your Loan Application Details")
        
        # Inputs in columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 **Income & Credit**")
            Applicant_Income = st.number_input("**Applicant Income (₹)**", min_value=0, value=300000, 
                                             help="Your annual salary")
            Coapplicant_Income = st.number_input("**Co-applicant Income (₹)**", min_value=0, value=100000)
            Credit_Score = st.number_input("**Credit Score**", min_value=300, max_value=900, value=650)
            Age = st.number_input("**Age**", min_value=18, max_value=100, value=30)
        
        with col2:
            st.subheader("👨‍👩‍👧 **Personal**")
            Employment_Status = st.selectbox("**Employment**", encoders["Employment_Status"].classes_)
            Marital_Status = st.selectbox("**Marital Status**", encoders["Marital_Status"].classes_)
            Education_Level = st.selectbox("**Education**", encoders["Education_Level"].classes_)
            Gender = st.selectbox("**Gender**", encoders["Gender"].classes_)
            Dependents = st.number_input("**Dependents**", min_value=0, max_value=10, value=1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏠 **Financials**")
            Existing_Loans = st.number_input("**Existing Loans**", min_value=0, max_value=10, value=0)
            DTI_Ratio = st.number_input("**DTI Ratio**", min_value=0.0, max_value=1.0, value=0.3, format="%.2f")
            Savings = st.number_input("**Savings (₹)**", min_value=0, value=50000)
        
        with col2:
            st.subheader("💳 **Loan Details**")
            Loan_Amount = st.number_input("**Loan Amount (₹)**", min_value=0, value=500000)
            Loan_Term = st.number_input("**Loan Term (months)**", min_value=12, value=60)
            Collateral_Value = st.number_input("**Collateral (₹)**", min_value=0, value=700000)
        
        col1, col2 = st.columns(2)
        with col1:
            Loan_Purpose = st.selectbox("**Loan Purpose**", encoders["Loan_Purpose"].classes_)
            Property_Area = st.selectbox("**Property Area**", encoders["Property_Area"].classes_)
        with col2:
            Employer_Category = st.selectbox("**Employer Category**", encoders["Employer_Category"].classes_)
        
        # Predict button
        if st.button("🚀 **PREDICT & GET ADVICE**", type="primary", use_container_width=True):
            with st.spinner("🤖 AI Analyzing your application..."):
                # Create input data
                input_data = pd.DataFrame({col: [0] for col in columns})
                
                input_data['Applicant_Income'] = Applicant_Income
                input_data['Coapplicant_Income'] = Coapplicant_Income
                input_data['Employment_Status'] = encoders["Employment_Status"].transform([Employment_Status])[0]
                input_data['Age'] = Age
                input_data['Marital_Status'] = encoders["Marital_Status"].transform([Marital_Status])[0]
                input_data['Dependents'] = Dependents
                input_data['Credit_Score'] = Credit_Score
                input_data['Existing_Loans'] = Existing_Loans
                input_data['DTI_Ratio'] = DTI_Ratio
                input_data['Savings'] = Savings
                input_data['Collateral_Value'] = Collateral_Value
                input_data['Loan_Amount'] = Loan_Amount
                input_data['Loan_Term'] = Loan_Term
                input_data['Loan_Purpose'] = encoders["Loan_Purpose"].transform([Loan_Purpose])[0]
                input_data['Property_Area'] = encoders["Property_Area"].transform([Property_Area])[0]
                input_data['Education_Level'] = encoders["Education_Level"].transform([Education_Level])[0]
                input_data['Gender'] = encoders["Gender"].transform([Gender])[0]
                input_data['Employer_Category'] = encoders["Employer_Category"].transform([Employer_Category])[0]
                
                # Scale and predict
                input_data[columns] = scaler.transform(input_data[columns])
                prob = model.predict_proba(input_data)[0][1]
                pred = 1 if prob > 0.5 else 0
                
                # Results display
                col1, col2 = st.columns([3,1])
                with col1:
                    if pred == 1:
                        st.markdown(f"""
                        <div class="metric-approved">
                            <h2>✅ LOAN APPROVED!</h2>
                            <h1 style='font-size: 3rem;'>{prob:.1%}</h1>
                            <p>Approval Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-rejected">
                            <h2>❌ LOAN REJECTED</h2>
                            <h1 style='font-size: 3rem;'>{prob:.1%}</h1>
                            <p>Approval Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Personalized advice
                top3 = importance.head(3)
                total_income = Applicant_Income + Coapplicant_Income
                assets_coverage = Collateral_Value / Loan_Amount if Loan_Amount > 0 else 0
                
                advice = f"""
                💡 **YOUR IMPROVEMENT PLAN** 
                
                🎯 **TOP 3 FACTORS** (AI Analysis):
                1. **{top3.iloc[0]['feature']}** ({top3.iloc[0]['importance']:.0%})
                2. **{top3.iloc[1]['feature']}** ({top3.iloc[1]['importance']:.0%})
                3. **{top3.iloc[2]['feature']}** ({top3.iloc[2]['importance']:.0%})
                
                📊 **YOUR STATUS vs TARGET**:
                • Credit Score: **{Credit_Score}** → Target **>750**
                • Total Income: **₹{total_income:,}** → Target **>₹8L**
                • DTI Ratio: **{DTI_Ratio:.1%}** → Target **<25%**
                • Assets Coverage: **{assets_coverage:.0%}** → Target **>120%**
                """
                st.markdown(f'<div class="advice-box">{advice}</div>', unsafe_allow_html=True)
                
                # Save to history
                result = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
                add_history(st.session_state.username, result, prob, advice)
                st.success("💾 Saved to your history!")
    
    elif menu == "📊 View History":
        st.subheader("📋 Your Prediction History")
        history = view_history(st.session_state.username)
        if history:
            df_hist = pd.DataFrame(history, columns=["Result", "Probability", "Advice", "Date"])
            df_hist["Probability"] = df_hist["Probability"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            st.info(f"🕒 Showing last 10 predictions")
        else:
            st.info("📝 No predictions yet. Try getting a loan prediction!")
    
    elif menu == "⚡ Key Factors":
        st.subheader("🔍 AI Decision Factors")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(importance.head(10).set_index('feature')['importance'])
        with col2:
            for i, row in importance.head(5).iterrows():
                st.metric(row['feature'], f"{row['importance']:.0%}")
    
    elif menu == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("👋 Logged out!")
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("*🎓 College Final Year Project - Smart Loan Approval AI*")