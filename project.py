import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Student Stress Dashboard", layout="wide")
st.markdown("<h1 style='text-decoration: underline;'>Student Workload & Stress Dashboard</h1>", unsafe_allow_html=True)

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("workload vs stress.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --------------------------
# Formal Names Mapping
# --------------------------
formal_names = {
    "Study_yr": "Year of Study",
    "stress_lv": "Stress Level (1-5)",
    "age": "Age",
    "cgpa": "CGPA",
    "study_hr(h/w)": "Study Hours per Week",
    "credits_now": "Credits Taken",
    "extra_curricular": "Extracurricular Activities",
    "curricular(h/w)": "Curricular Hours per Week",
    "part_time": "Part-time Job",
    "part_time(h/w)": "Part-time Hours per Week",
    "avg_sleep(h/n)": "Average Sleep Hours",
    "sleep_quality": "Sleep Quality (1-10)",
    "anxiety_level": "Anxiety Level (1-5)",
    "gender_female": "Female",
    "gender_male": "Male",
    "gender_unknown": "Unknown"
}
df.rename(columns=formal_names, inplace=True)

# --------------------------
# Convert 0/1 columns to Yes/No
# --------------------------
df["Part-time Job"] = df["Part-time Job"].apply(lambda x: "Yes" if x == 1 else "No")
df["Curricular Activity"] = df["Curricular Hours per Week"].apply(lambda x: "Yes" if x > 0 else "No")

# --------------------------
# Sidebar Filters
# --------------------------
with st.sidebar:
    st.markdown("<h3 style='text-decoration: underline;'>Filters</h3>", unsafe_allow_html=True)
    st.caption("▸ Refine the dataset using the filters below")

    years = st.multiselect("Year of Study", sorted(df["Year of Study"].unique()), default=sorted(df["Year of Study"].unique()))
    genders = st.multiselect("Gender", ["Female", "Male", "Unknown"], default=["Female", "Male", "Unknown"])
    jobs = st.multiselect("Part-time Job", ["Yes", "No"], default=["Yes", "No"])
    curricular = st.multiselect("Curricular Activity", ["Yes", "No"], default=["Yes", "No"])

# Apply filters
gender_mask = pd.Series(False, index=df.index)
if "Female" in genders: gender_mask |= df["Female"]
if "Male" in genders: gender_mask |= df["Male"]
if "Unknown" in genders: gender_mask |= df["Unknown"]

df_f = df[
    df["Year of Study"].isin(years) &
    df["Part-time Job"].isin(jobs) &
    df["Curricular Activity"].isin(curricular) &
    gender_mask
]

# Handle empty dataset
if len(df_f) == 0:
    st.warning("No data available for the selected filters. Please adjust the filters.")
    st.stop()

# --------------------------
# Overview Metrics
# --------------------------
st.markdown("<h3 style='text-decoration: underline;'>Overview Metrics</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Number of Students", len(df_f))
c2.metric("Average Study Hours", f"{df_f['Study Hours per Week'].mean():.1f}")
c3.metric("Average Sleep Hours", f"{df_f['Average Sleep Hours'].mean():.1f}")
c4.metric("Average Stress Level", f"{df_f['Stress Level (1-5)'].mean():.2f}")

# --------------------------
# Target Variable Selection
# --------------------------
st.markdown("<h3 style='text-decoration: underline;'>Target Variable Selection</h3>", unsafe_allow_html=True)
dependent_vars = ["Stress Level (1-5)", "Anxiety Level (1-5)", "Sleep Quality (1-10)", "Average Sleep Hours"]
target_var = st.selectbox("Choose Target Variable (Dependent Variable)", options=dependent_vars)

# --------------------------
# Feature Selection for Plots
# --------------------------
numeric_cols = df_f.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_var in numeric_cols:
    numeric_cols.remove(target_var)

st.markdown("<h3 style='text-decoration: underline;'>Feature Selection for Plots</h3>", unsafe_allow_html=True)
x_feature = st.selectbox("Select X Feature", options=numeric_cols, index=0)
y_feature = target_var

# --------------------------
# Scatter Plot
# --------------------------
st.markdown(f"<h3 style='text-decoration: underline;'>Scatter Plot: {y_feature} vs {x_feature}</h3>", unsafe_allow_html=True)
fig = px.scatter(df_f, x=x_feature, y=y_feature, color="Year of Study", trendline="ols", hover_data=df_f.columns)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Boxplots for categorical variables
# --------------------------
st.markdown(f"<h3 style='text-decoration: underline;'>Boxplot: {y_feature} by Year of Study</h3>", unsafe_allow_html=True)
fig2 = px.box(df_f, x="Year of Study", y=y_feature, points="all", color="Year of Study")
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# Correlation Heatmap
# --------------------------
st.markdown("<h3 style='text-decoration: underline;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
corr_cols = st.multiselect("Select Numeric Columns for Correlation", options=numeric_cols + [target_var], default=[x_feature, target_var])
if len(corr_cols) >= 2:
    corr_matrix = df_f[corr_cols].corr()
    fig5 = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='Viridis')
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Select at least 2 numeric columns for correlation heatmap.")

# --------------------------
# Multi-feature Regression
# --------------------------
st.markdown("<h3 style='text-decoration: underline;'>Advanced Regression: Multi-feature Predictor</h3>", unsafe_allow_html=True)
selected_features = st.multiselect("Select Features for Regression (X)", options=numeric_cols, default=numeric_cols[:3])
models = {}
if len(selected_features) >= 1:
    X_multi = df_f[selected_features].values
    y_multi = df_f[target_var].values
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1)
    }
    
    for name, model in models.items():
        model.fit(X_multi, y_multi)
        y_pred = model.predict(X_multi)
        st.markdown(f"<h4>► {name}</h4>", unsafe_allow_html=True)
        coef_dict = {feat: coef for feat, coef in zip(selected_features, model.coef_)}
        st.write(f"Intercept: {model.intercept_:.3f}")
        st.write("Coefficients:", coef_dict)
        st.write(f"R² Score: {r2_score(y_multi, y_pred):.3f}")

        # Predicted vs Actual plot
        st.markdown(f"<h5>▸ Predicted vs Actual ({name})</h5>", unsafe_allow_html=True)
        df_pred_multi = pd.DataFrame({"Actual": y_multi, "Predicted": y_pred})
        fig_pred_multi = px.scatter(df_pred_multi, x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig_pred_multi, use_container_width=True)
else:
    st.info("Select at least 1 feature for advanced regression.")

# --------------------------
# Detailed Interpretation Panel
# --------------------------
st.markdown("<h3 style='text-decoration: underline;'>Detailed Regression Interpretation</h3>", unsafe_allow_html=True)
if len(selected_features) >= 1:
    lin_model = models["Linear Regression"]
    y_pred = lin_model.predict(X_multi)
    r2 = r2_score(y_multi, y_pred)

    st.markdown("<h4>► Feature Effects on Target Variable</h4>", unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": lin_model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    for _, row in coef_df.iterrows():
        direction = "increases" if row["Coefficient"] > 0 else "decreases"
        st.write(f"• <b>{row['Feature']}</b> {direction} <b>{target_var}</b> by {abs(row['Coefficient']):.3f} per unit change", unsafe_allow_html=True)

    st.markdown("<h4>► Model Fit</h4>", unsafe_allow_html=True)
    st.write(f"— R²: **{r2:.3f}**")

    if r2 < 0.1:
        st.warning("The model explains very little variability. Likely missing key variables or non-linear effects.")
    elif r2 < 0.3:
        st.warning("The model explains only a small portion of variability. Consider additional features or non-linear modeling.")
    elif r2 < 0.5:
        st.info("The model explains a modest portion of variability. Additional factors may be relevant.")
    elif r2 < 0.7:
        st.info("The model moderately explains variability. Linear relationships are present but incomplete.")
    elif r2 < 0.9:
        st.success("The model explains a large portion of variability. Strong linear relationships are evident.")
    else:
        st.success("The model explains almost all variability. Excellent fit, though overfitting should be checked.")

    st.markdown("<h4>► Notes on Model Limitations</h4>", unsafe_allow_html=True)
    for feat in selected_features:
        coef_sign = np.sign(lin_model.coef_[selected_features.index(feat)])
        correlation = df_f[[feat, target_var]].corr().iloc[0, 1]
        if abs(correlation) < 0.3:
            st.write(f"• <i>{feat}</i> may not capture {target_var} effectively (weak linear correlation).", unsafe_allow_html=True)
        elif np.sign(correlation) != coef_sign:
            st.write(f"• <i>{feat}</i> shows a coefficient direction opposite to its correlation, suggesting non-linear effects.", unsafe_allow_html=True)

    st.info("— Consider non-linear or ensemble models if multiple features show weak or inconsistent linearity.")
else:
    st.info("Select at least 1 feature to see detailed interpretation.")

# --------------------------
# Show Filtered Data
# --------------------------
with st.expander("Show Filtered Data"):
    st.dataframe(df_f)
    st.download_button("Download Filtered Data as CSV", df_f.to_csv(index=False), "filtered_data.csv", "text/csv", key='download-csv')