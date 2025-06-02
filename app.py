import streamlit as st
import pandas as pd
import joblib
import sqlite3
from hashlib import sha256

# --------- DATABASE SETUP --------------

conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

def create_user_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()

def add_user(username, password):
    password_hash = sha256(password.encode()).hexdigest()
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

def verify_user(username, password):
    password_hash = sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', (username, password_hash))
    return c.fetchone() is not None

create_user_table()

st.set_page_config(page_title="Symptom-Disease Predictor", layout="wide")

# ---------- Load model & data -----------

@st.cache_data
def load_data():
    return pd.read_excel("dataset.xlsx")

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("disease_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

df = load_data()
model, vectorizer = load_model_and_vectorizer()

# --------- AUTHENTICATION PAGES ------------

def register():
    st.title("📝 Create a New Account")
    new_user = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type="password")
    new_password_confirm = st.text_input("Confirm password", type="password")

    if st.button("Register"):
        if not new_user or not new_password or not new_password_confirm:
            st.warning("Please fill all fields.")
        elif new_password != new_password_confirm:
            st.error("Passwords do not match.")
        else:
            success = add_user(new_user, new_password)
            if success:
                st.success("User registered successfully! Please login.")
                st.session_state['page'] = "login"
            else:
                st.error("Username already exists. Please choose another.")

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['page'] = "login"

# --------- MAIN APP -----------

def symptom_predictor():
    st.title("🩺 Intelligent Symptom-to-Disease Predictor")
    st.markdown(f"Welcome **{st.session_state['username']}**, enter your symptoms below:")
    user_input = st.text_area("📝 Enter symptoms (comma-separated):", "fever, cough, sore throat")

    if st.button("🔍 Predict Disease"):
        if user_input.strip():
            try:
                input_tfidf = vectorizer.transform([user_input])
                probs = model.predict_proba(input_tfidf)[0]
                top_indices = probs.argsort()[-5:][::-1]
                top_diseases = model.classes_[top_indices]
                top_probs = probs[top_indices]

                st.markdown("### Top 5 Predicted Diseases:")

                for i, disease in enumerate(top_diseases):
                    row = df[df['disease'] == disease].iloc[0]
                    risk = row['risk level']
                    doctor = row['doctor']
                    st.markdown(f"""
                        **{i+1}. {disease}**  
                        - 🔥 Risk Level: `{risk}`  
                        - 👨‍⚕️ Doctor Specialties: `{doctor}`  
                        - 📊 Confidence: `{top_probs[i]*100:.2f}%`
                    """)
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
        else:
            st.warning("⚠️ Please enter symptoms to get a prediction.")

    st.markdown("---")
    if st.button("Logout"):
        logout()

    st.caption("❤️Recover, restore, and reclaim your health.")

# ---------- APP PAGE CONTROLLER ---------

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = "login"
if 'username' not in st.session_state:
    st.session_state['username'] = None

if st.session_state['logged_in']:
    symptom_predictor()
else:
    if st.session_state['page'] == "login":
        login()
        st.markdown("---")
        if st.button("Create new account"):
            st.session_state['page'] = "register"
    elif st.session_state['page'] == "register":
        register()
        st.markdown("---")
        if st.button("Back to login"):
            st.session_state['page'] = "login"
