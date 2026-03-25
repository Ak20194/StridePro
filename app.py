import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="RunFit AI: Data-Driven Startup", layout="wide")

# --- DATA GENERATION WITH NOISE & OUTLIERS ---
@st.cache_data
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    
    # 1. Demographics
    ages = np.random.randint(18, 65, n)
    emirates = np.random.choice(['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman', 'RAK'], n, p=[0.45, 0.35, 0.1, 0.05, 0.05])
    occupations = np.random.choice(['Student', 'Corporate', 'Healthcare', 'Entrepreneur'], n)
    
    # 2. Logic-Driven Features
    mileage = np.random.randint(5, 75, n)
    # Regression Target: Budget (Influenced by Age and Occupation)
    budget = 200 + (ages * 8) + np.random.normal(0, 100, n)
    budget = np.where(occupations == 'Entrepreneur', budget * 1.4, budget)
    
    # 3. Preferences (For Association Rules)
    colors = np.random.choice(['Neon', 'Classic Black', 'Earth Tones'], n)
    soles = np.random.choice(['Carbon-Plated', 'Max Cushion', 'Minimalist'], n)
    
    # 4. Target Label: App Interest (Classification)
    # Probability increases with mileage and budget
    prob = (mileage/75 * 0.5) + (budget/1500 * 0.5)
    app_interest = [1 if p > 0.55 else 0 for p in prob]
    
    # 5. Injecting Noise & Outliers (The "Messy" Data)
    budget[0:15] = 4500  # High-end outliers
    budget[15:30] = 20    # Data entry errors / Noise
    
    df = pd.DataFrame({
        'Age': ages,
        'Emirate': emirates,
        'Occupation': occupations,
        'Weekly_KM': mileage,
        'Shoe_Budget_AED': budget.astype(int),
        'Preferred_Color': colors,
        'Sole_Technology': soles,
        'Interested_in_App': app_interest
    })
    return df

df = generate_synthetic_data()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Project Overview", "Data Export", "Classification", "Clustering", "Association Rules", "Regression"])

# --- MODULE: DATA EXPORT ---
if menu == "Data Export":
    st.title("📂 Data Generation & CSV Export")
    st.write("Below is the synthetic dataset of 1,000 respondents with built-in patterns and noise.")
    st.dataframe(df.head(20))
    
    # CSV Conversion Logic
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Dataset (1,000 Rows)",
        data=csv,
        file_name='runfit_synthetic_data.csv',
        mime='text/csv',
    )
    st.info("Download this file to use in external tools like PowerBI or Excel.")

# --- MODULE: CLASSIFICATION ---
elif menu == "Classification":
    st.title("🎯 Classification Performance")
    X = pd.get_dummies(df[['Age', 'Weekly_KM', 'Shoe_Budget_AED']])
    y = df['Interested_in_App']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics Table
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    m4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
    
    # ROC Curve & Importance
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC: {auc(fpr, tpr):.2f})")
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)
    
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    st.plotly_chart(px.bar(feat_imp, orientation='h', title="Feature Importance"), use_container_width=True)

# --- MODULE: ASSOCIATION RULES ---
elif menu == "Association Rules":
    st.title("🔗 Product Association Mining")
    basket = pd.get_dummies(df[['Preferred_Color', 'Sole_Technology']])
    frequent = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    
    st.write("Discovering connections between shoe styles and tech preferences:")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))

# (Rest of modules like Clustering and Regression would follow similar logic)