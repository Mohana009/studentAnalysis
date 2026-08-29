# =====================================================================
# app.py
# Student Academic Risk Intelligence System - Streamlit Dashboard
# =====================================================================

# --- Library imports ---
import streamlit as st               # For building the interactive dashboard
import pandas as pd                  # For loading and manipulating the dataset
import plotly.express as px          # For interactive charts (used later)


# =====================================================================
# Step 1: Set Streamlit page configuration
# =====================================================================
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓"
)


# =====================================================================
# Step 2: Data loading + feature engineering function
# =====================================================================
@st.cache_data  # Cache the result so the CSV isn't reloaded/reprocessed on every rerun
def load_and_prepare_data():
    """
    Loads Maths.csv from the data/ folder and applies the same feature
    engineering used across analysis.py and main.py.

    Returns:
        pd.DataFrame: The dataset with all engineered columns added
    """

    # Load the raw CSV file from the data/ folder
    data = pd.read_csv("data/Maths.csv")

    # --- Feature: Result ---
    # G3 = 0 -> Dropout, G3 1-9 -> Fail, G3 10-20 -> Pass
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    data["Result"] = data["G3"].apply(classify_result)

    # --- Feature: Percentage ---
    # Convert final grade (out of 20) into a percentage
    data["Percentage"] = (data["G3"] / 20) * 100

    # --- Feature: avg_alcohol ---
    # Average of weekday (Dalc) and weekend (Walc) alcohol consumption
    data["avg_alcohol"] = (data["Dalc"] + data["Walc"]) / 2

    # --- Feature: parent_edu_avg ---
    # Average of mother's (Medu) and father's (Fedu) education level
    data["parent_edu_avg"] = (data["Medu"] + data["Fedu"]) / 2

    # --- Feature: grade_trend ---
    # Difference between final grade and first period grade
    data["grade_trend"] = data["G3"] - data["G1"]

    # --- Feature: total_support ---
    # Count of "yes" values across schoolsup, famsup, and paid columns
    support_cols = ["schoolsup", "famsup", "paid"]
    data["total_support"] = (data[support_cols] == "yes").sum(axis=1)

    # --- Feature: risk_score ---
    # Composite score combining failures, absences, alcohol use, and study time
    data["risk_score"] = (
        (data["failures"] * 2)
        + (data["absences"] / 10)
        + data["avg_alcohol"]
        - data["studytime"]
    )

    # --- Feature: g1_g2_avg ---
    # Average of first and second period grades
    data["g1_g2_avg"] = (data["G1"] + data["G2"]) / 2

    # Return the fully prepared DataFrame
    return data


# Load the prepared data once (cached across reruns)
df = load_and_prepare_data()


# =====================================================================
# Step 3: Main dashboard title
# =====================================================================
st.title("🎓 Student Academic Risk Intelligence System")


# =====================================================================
# Step 4: KPI metric cards (4 cards in one row)
# =====================================================================

# Filter out dropouts (G3 = 0) for grade-based calculations
non_dropout_df = df[df["G3"] != 0]

# KPI 1: Total number of students in the dataset
total_students = len(df)

# FIX: guard against an empty non_dropout_df (e.g. every student dropped
# out, or an empty/filtered dataset). Without this, .mean() would return
# NaN and the pass-rate division would raise ZeroDivisionError, which
# would crash the entire app on load since Streamlit runs top-to-bottom
# and stops on the first unhandled exception.
if len(non_dropout_df) > 0:
    # KPI 2: Class average G3, excluding dropouts, rounded to 2 decimals
    class_average_g3 = round(non_dropout_df["G3"].mean(), 2)

    # KPI 3: Pass rate % among non-dropout students, rounded to 1 decimal
    # NOTE: rounded to 1 decimal here (vs 2 decimals for class_average_g3)
    # intentionally, per spec — not a typo/inconsistency.
    pass_count = (non_dropout_df["G3"] >= 10).sum()
    pass_rate_percent = round((pass_count / len(non_dropout_df)) * 100, 1)
else:
    class_average_g3 = 0.0
    pass_rate_percent = 0.0

# KPI 4: At-risk count -> students who failed but did not drop out (G3 1-9)
at_risk_count = ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()

# Create 4 equal-width columns to display the KPI cards in one row
col1, col2, col3, col4 = st.columns(4)

# Display each KPI using Streamlit's metric widget
with col1:
    st.metric(label="Total Students", value=total_students)

with col2:
    st.metric(label="Class Average G3", value=class_average_g3)

with col3:
    st.metric(label="Pass Rate %", value=f"{pass_rate_percent}%")

with col4:
    st.metric(label="At-Risk Count", value=int(at_risk_count))


# =====================================================================
# Section: Performance Charts (side by side)
# =====================================================================
st.subheader("📊 Performance Charts")

# Create 2 equal-width columns to display the charts side by side
chart_col1, chart_col2 = st.columns(2)

# ---------------------------------------------------------------
# Left Chart: Scatter plot - Study Time vs Final Grade
# ---------------------------------------------------------------
with chart_col1:

    # Fixed color mapping so results are visually consistent
    result_color_map = {
        "Pass": "green",
        "Fail": "red",
        "Dropout": "grey"
    }

    # Build the scatter plot using Plotly Express
    scatter_fig = px.scatter(
        df,
        x="studytime",                        # X axis: study time level
        y="G3",                               # Y axis: final grade
        color="Result",                       # Color points by Pass/Fail/Dropout
        color_discrete_map=result_color_map,  # Apply fixed color scheme
        hover_data=["absences", "G1", "G2"],  # Extra info shown on hover
        title="Study Time vs Final Grade"
    )

    # Render the chart in Streamlit, expanding to fill the column width
    st.plotly_chart(scatter_fig, use_container_width=True)

# ---------------------------------------------------------------
# Right Chart: Bar chart - Average G3 by Internet Access
# ---------------------------------------------------------------
with chart_col2:

    # Group by internet access (yes/no) and compute mean G3 for each group
    avg_g3_by_internet = df.groupby("internet", as_index=False)["G3"].mean()

    # Build the bar chart using Plotly Express
    bar_fig = px.bar(
        avg_g3_by_internet,
        x="internet",           # X axis: internet access (yes/no)
        y="G3",                 # Y axis: average final grade
        title="Average G3 by Internet Access"
    )

    # Render the chart in Streamlit, expanding to fill the column width
    st.plotly_chart(bar_fig, use_container_width=True)


# =====================================================================
# Section: Student Analysis Table
# =====================================================================
st.subheader("🚨 Student Analysis Table")

# Step 1: Dropdown to filter students by Result category
result_filter = st.selectbox(
    label="Filter by Result",
    options=["All", "Pass", "Fail", "Dropout"]
)

# Step 2: Filter the DataFrame based on the dropdown selection
if result_filter == "All":
    # No filtering needed, show every student
    filtered_df = df
else:
    # Show only students matching the selected Result category
    # NOTE: when filtering to "Dropout", G3 and Percentage will always
    # show as 0 / 0.0 for every row in this table. That's expected per
    # the business rule (G3=0 means Dropout, not a real zero score) —
    # it isn't a bug, just something that can visually read as "these
    # students scored 0%" if not kept in mind.
    filtered_df = df[df["Result"] == result_filter]

# Step 3: Display the filtered DataFrame with only the required columns
table_columns = [
    "G1", "G2", "G3", "Result", "Percentage",
    "absences", "studytime", "failures", "risk_score"
]

st.dataframe(filtered_df[table_columns], use_container_width=True)


# =====================================================================
# Sub-section: At-Risk Students
# =====================================================================
st.subheader("⚠️ At-Risk Students")

# NOTE: this table intentionally always shows ALL at-risk students from
# the full dataset — it does NOT respect the "Filter by Result" dropdown
# above. The two sections are independent by design.

# Filter for at-risk students: G3 between 1 and 9 inclusive (failed, not dropped out)
at_risk_df = df[(df["G3"] >= 1) & (df["G3"] <= 9)].copy()

# Sort by G3 ascending so worst-performing students appear first
at_risk_df = at_risk_df.sort_values(by="G3", ascending=True)

# Select only the required columns for this table
at_risk_columns = ["G1", "G2", "G3", "absences", "studytime", "failures"]

# Display the at-risk students table
st.dataframe(at_risk_df[at_risk_columns], use_container_width=True)

# Show the total count of at-risk students
st.write(f"Total at-risk students: {len(at_risk_df)}")