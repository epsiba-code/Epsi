import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Page Configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# --- CENTRALIZED GLOBAL UI/UX STYLING PACK ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2 family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Master Deep Slate Blue Professional Background Scheme */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0D9488 100%);
    color: #E2E8F0;
}

/* Centering Logic for the Auth Screen Wrapper Container */
div[data-testid="stVerticalBlock"] > div:has(div.auth-container) {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

/* Attractive Centered Form Card Wrapper */
.auth-container {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    border: 2px solid #38BDF8;
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    max-width: 520px;
    margin: 40px auto;
    text-align: center;
}

/* Navigation Menu Customizations */
section[data-testid="stSidebar"] {
    background: #0B0F19 !important;
    border-right: 1px solid #1E293B;
}

section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Professional Soft Gold and Ice Blue Typography Settings */
h1, h2 {
    color: #FCD34D !important; /* Soft Gold Accent */
    font-weight: 700;
}

h3, h4, label, p, .stMarkdown {
    color: #E0F2FE !important; /* Premium Ice Blue Contrast */
}

/* Professional Interactive Input Controls */
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

/* Unified Action Buttons styling */
.stButton>button {
    background: linear-gradient(90deg, #F59E0B, #D97706);
    color: #0F172A !important;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-weight: 700;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.3);
}

.stButton>button:hover {
    background: linear-gradient(90deg, #38BDF8, #0284C7);
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
}

/* Interactive Tabs adjustments */
button[data-baseweb="tab"] {
    color: #94A3B8 !important;
}
button[aria-selected="true"] {
    color: #FCD34D !important;
    border-bottom-color: #FCD34D !important;
}

/* Metric Display Panels */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid #334155 !important;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FCD34D !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []
if "data" not in st.session_state:
    st.session_state.data = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "models" not in st.session_state:
    st.session_state.models = {}
if "users" not in st.session_state:
    st.session_state.users = {
        "admin": "password123"
    }

# --- LOGIN / LOGOUT LOGIC ---
def login(username, password):
    users = st.session_state.users

    if username in users and users[username] == password:
        st.session_state.logged_in = True
        st.success("Login Successful!")
        st.rerun()
    else:
        st.error("Invalid Username or Password")
def logout():
    st.session_state.logged_in = False
    st.rerun()

# --- LOGIN PAGE ---
if not st.session_state.logged_in:

    st.title("🎓 Student Performance Predictor")

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    # LOGIN
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            login_btn = st.form_submit_button("Login")

            if login_btn:
                login(username, password)

    # REGISTER
    with tab2:
        with st.form("register_form"):

            new_username = st.text_input("Create Username")
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            register_btn = st.form_submit_button("Register")

            if register_btn:

                if new_username in st.session_state.users:
                    st.error("Username already exists!")

                elif new_password != confirm_password:
                    st.error("Passwords do not match!")

                elif new_username == "" or new_password == "":
                    st.error("Please fill all fields!")

                else:
                    st.session_state.users[new_username] = new_password
                    st.success("Registration Successful! Now login using your account.")
else:
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Upload Data", "Prediction", "Chatbot", "History"]
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("Logout 🚪"):
        logout()

    # Define features and target based on user requirement
    features = ['math_score', 'science_score', 'english_score', 'attendance_percentage', 'study_hours']
    target = 'overall_score'

    # --- 1. DASHBOARD ---
    if page == "Dashboard":
       
        st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Background Image */
.stApp{
    background:
        linear-gradient(rgba(255,255,255,0.92), rgba(255,255,255,0.92)),
        url("https://images.unsplash.com/photo-1524995997946-a1c2e315a42f")
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#ffffff;
    border-right:2px solid #dbeafe;
}

section[data-testid="stSidebar"] *{
    color:#1e3a8a;
}

/* Cards */
div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.15);
}

/* Headings */
h1,h2,h3{
    color:#2563eb;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(90deg,#3b82f6,#2563eb);
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#2563eb,#1d4ed8);
}

/* Input Boxes */
.stTextInput input,
.stNumberInput input{
    background:white;
    color:black;
    border:1px solid #cbd5e1;
    border-radius:8px;
}

/* Dataframe */
[data-testid="stDataFrame"]{
    background:white;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,.15);
}

</style>
""", unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            st.subheader("Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Students", len(df))
            col2.metric("Average Overall Score", round(df[target].mean(), 2))
            col3.metric("Average Attendance", f"{round(df['attendance_percentage'].mean(), 2)}%")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Quick Insights")
            st.line_chart(df[[target] + features[:3]].head(50))
        else:
            st.info("💡 Please go to the 'Upload Data' page to upload your dataset first.")

    # --- 2. UPLOAD DATA ---
    elif page == "Upload Data":
        st.markdown("""
        <h1 style='color:#38bdf8'>
        📂 Upload Student Dataset
        </h1>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your Student Performance CSV file", type=["csv"])
        
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Columns found in CSV:")
                st.write(df.columns.tolist())
                # Clean column names just in case there are trailing spaces
                df.columns = (
                   df.columns.str.strip()
                   .str.lower()
                   .str.replace(" ", "_")
                )

                # Remove rows with missing values
                df = df.dropna()
                
                # Check if required columns exist
                required_cols = [
                    "math_score",
                    "science_score",
                    "english_score",
                    "attendance_percentage",
                    "study_hours",
                    "overall_score"
                ]
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns in CSV: {missing_cols}")
                else:
                    # Convert all required columns to numeric
                    for col in required_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    df = df.dropna()

                    st.session_state.data = df
                    st.success("Dataset successfully uploaded and verified!")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Train Models Automatically upon Upload
                    st.write("⚙️ Training Models (Random Forest, XGBoost, SVM)...")
                    X = df[features].astype(float)
                    y = df[target].astype(float)
                    if len(df) < 10:
                        st.error("Dataset must contain at least 10 rows.")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42
                    )
                    
                    # Random Forest
                    rf = RandomForestRegressor(random_state=42)
                    rf.fit(X_train, y_train)
                    
                    # XGBoost
                    xgb = XGBRegressor(random_state=42)
                    xgb.fit(X_train, y_train)
                    
                    # SVM
                    svm = SVR()
                    svm.fit(X_train, y_train)
                    
                    # Store models in session state
                    st.session_state.models = {'Random Forest': rf, 'XGBoost': xgb, 'SVM': svm}
                    st.session_state.model_trained = True
                    st.success("All models trained successfully!")
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # --- 3. PREDICTION ---
    elif page == "Prediction":
        st.markdown("""
        <h1 style='color:#38bdf8'>
        🎯 Student Performance Prediction
        </h1>
        """, unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please upload a dataset in the 'Upload Data' tab to train the models first.")
        else:
            st.subheader("Enter Student Details:")
            
            col1, col2 = st.columns(2)
            with col1:
                math = st.number_input("Math Score", min_value=0.0, max_value=100.0, value=70.0)
                science = st.number_input("Science Score", min_value=0.0, max_value=100.0, value=75.0)
                english = st.number_input("English Score", min_value=0.0, max_value=100.0, value=80.0)
            with col2:
                attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0)
                study_h = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=4.5)
                algorithm = st.selectbox("Select ML Algorithm", ["Random Forest", "XGBoost", "SVM"])
            
            if st.button("Predict Overall Score"):
                input_df = pd.DataFrame([[math, science, english, attendance, study_h]], columns=features)
                selected_model = st.session_state.models[algorithm]
                
                # Predict
                prediction = selected_model.predict(input_df)[0]
                prediction = round(float(prediction), 2)
                # Grade Calculation
                if prediction >= 90:
                    grade = "A+"
                elif prediction >= 80:
                 grade = "A"
                elif prediction >= 70:
                    grade = "B"
                elif prediction >= 60:
                    grade = "C"
                elif prediction >= 50:
                    grade = "D"
                else:
                     grade = "F"

                # Pass / Fail
                status = "PASS ✅" if prediction >= 50 else "FAIL ❌"
                
                st.markdown(f"""
                <div style="
                background:linear-gradient(90deg,#10b981,#2563eb);
                padding:30px;
                border-radius:20px;
                text-align:center;
                color:white;
                ">

                <h2>Prediction Completed</h2>

                <h1>{prediction}</h1>

                <h3>Grade : {grade}</h3>

                <h3>Status : {status}</h3>

                <h4>Algorithm : {algorithm}</h4>

                </div>
                """, unsafe_allow_html=True)
                
                # Save to History
                history_entry = {
                    "Math Score": math,
                    "Science Score": science,
                    "English Score": english,
                    "Attendance %": attendance,
                    "Study Hours": study_h,
                    "Predicted Overall Score": prediction,
                    "Grade": grade,
                    "Status": status,
                    "Algorithm Used": algorithm
                }
                st.session_state.history.append(history_entry)

    # --- 4. CHATBOT ---
    elif page == "Chatbot":
        st.markdown("""
        <h1 style='color:#38bdf8'>
        🤖 AI Student Advisor
        </h1>

        <p style='color:white'>
        Ask anything related to studies and performance.
        </p>
        """, unsafe_allow_html=True)
        st.write("Ask anything about student performance improvement or study tips!")
        
        # Rule-based / Basic LLM placeholder setup
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your student advisor. How can I help you today?"}]
            
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if user_query := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            
            # Simple custom responses based on keywords (can be connected to OpenAI API)
            response = "That's a good question! To improve overall scores, focus on consistent study hours and maintaining attendance above 85%."
            if "math" in user_query.lower():
                response = "For Math, focus on practicing formula-based problems daily and analyzing weak areas."
            elif "attendance" in user_query.lower():
                response = "Attendance has a direct impact on scores. Try not to miss classes to ensure continuity in learning."
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

    # --- 5. HISTORY ---
    elif page == "History":
        st.markdown("""
        <h1 style='color:#38bdf8'>
        📜 Prediction History
        </h1>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.history) == 0:
            st.info("No predictions made yet.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun() 