# =====================================================================
# main.py
# Student Academic Risk Intelligence System - FastAPI Setup
# =====================================================================

# --- Core FastAPI imports ---
from fastapi import FastAPI          # Main FastAPI class to create the app
from pydantic import BaseModel, Field  # BaseModel for schemas, Field for validation rules

# --- Data processing imports ---
import pandas as pd                  # For loading and manipulating the dataset
import numpy as np                   # For numerical calculations (used in feature engineering)

# --- Server import ---
import uvicorn                       # ASGI server to run the FastAPI app


# =====================================================================
# Step 1: Create the FastAPI application instance
# =====================================================================
app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0"
)


# =====================================================================
# Step 2: Data loading + feature engineering function
# =====================================================================
def load_data():
    """
    Loads the Maths.csv dataset from the data/ folder and applies the
    same feature engineering used in analysis.py, so the API works with
    a fully prepared DataFrame.

    Returns:
        pd.DataFrame: The dataset with all engineered columns added
    """

    # Load the raw CSV file from the data/ folder
    # NOTE: this runs at import time (see df = load_data() below), so if
    # the CSV is missing or the working directory is wrong, the app will
    # fail to start entirely rather than returning a runtime 500 error.
    # This is a deliberate "fail fast" choice for this project.
    data = pd.read_csv("data/Maths.csv")

    # --- Feature 1: Result ---
    # Classify each student based on final grade (G3):
    # G3 = 0 -> Dropout (not a real score of zero)
    # G3 1-9 -> Fail
    # G3 10-20 -> Pass
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    data["Result"] = data["G3"].apply(classify_result)

    # --- Feature 2: Percentage ---
    # Convert final grade (out of 20) into a percentage
    data["Percentage"] = (data["G3"] / 20) * 100

    # --- Feature 3: avg_alcohol ---
    # Average of weekday (Dalc) and weekend (Walc) alcohol consumption
    data["avg_alcohol"] = (data["Dalc"] + data["Walc"]) / 2

    # --- Feature 4: parent_edu_avg ---
    # Average of mother's (Medu) and father's (Fedu) education level
    data["parent_edu_avg"] = (data["Medu"] + data["Fedu"]) / 2

    # --- Feature 5: grade_trend ---
    # Difference between final grade and first period grade
    # Positive = improved over the year, Negative = declined
    data["grade_trend"] = data["G3"] - data["G1"]

    # --- Feature 6: total_support ---
    # Count of "yes" values across schoolsup, famsup, and paid columns
    support_cols = ["schoolsup", "famsup", "paid"]
    data["total_support"] = (data[support_cols] == "yes").sum(axis=1)

    # --- Feature 7: risk_score ---
    # Composite score combining failures, absences, alcohol use, and study time
    # Higher failures/absences/alcohol -> higher risk
    # Higher studytime -> lower risk (subtracted)
    data["risk_score"] = (
        (data["failures"] * 2)
        + (data["absences"] / 10)
        + data["avg_alcohol"]
        - data["studytime"]
    )

    # --- Feature 8: g1_g2_avg ---
    # Average of first and second period grades
    data["g1_g2_avg"] = (data["G1"] + data["G2"]) / 2

    # Return the fully prepared DataFrame
    return data


# =====================================================================
# Step 3: Load the dataset once at startup and store it globally
# =====================================================================
# This avoids reloading/reprocessing the CSV on every API request.
df = load_data()


# =====================================================================
# Endpoint 1: GET /summary
# Returns overall class statistics (excluding dropouts for averages)
# =====================================================================
@app.get("/summary")
def get_summary():
    """
    Returns a JSON summary of the class:
    - total_students
    - class_average_g3 (non-dropouts only)
    - pass_rate_percent (non-dropouts only)
    - at_risk_count (G3 between 1 and 9)
    - dropout_count (G3 = 0)
    """

    # Total number of students in the dataset
    total_students = len(df)

    # Filter out dropouts (G3 = 0) since their grade isn't a real score
    non_dropout_df = df[df["G3"] != 0]

    # FIX: guard against an empty non_dropout_df (e.g. every student
    # dropped out, or an empty/filtered dataset) to avoid ZeroDivisionError
    # and NaN values that would break JSON serialization.
    if len(non_dropout_df) > 0:
        # Class average G3, calculated only from non-dropout students
        # float(...) ensures a native Python float, not numpy.float64
        class_average_g3 = float(round(non_dropout_df["G3"].mean(), 2))

        # Pass rate: % of non-dropout students with G3 >= 10
        pass_count = (non_dropout_df["G3"] >= 10).sum()
        pass_rate_percent = float(round((pass_count / len(non_dropout_df)) * 100, 2))
    else:
        class_average_g3 = 0.0
        pass_rate_percent = 0.0

    # At-risk count: students who failed but did not drop out (G3 1-9)
    at_risk_count = int(((df["G3"] >= 1) & (df["G3"] <= 9)).sum())

    # Dropout count: students with G3 = 0
    dropout_count = int((df["G3"] == 0).sum())

    # Return everything as a JSON-serializable dictionary
    return {
        "total_students": total_students,
        "class_average_g3": class_average_g3,
        "pass_rate_percent": pass_rate_percent,
        "at_risk_count": at_risk_count,
        "dropout_count": dropout_count
    }


# =====================================================================
# Endpoint 2: GET /at-risk
# Returns students who failed (G3 between 1 and 9), worst first
# =====================================================================
@app.get("/at-risk")
def get_at_risk_students():
    """
    Returns a list of at-risk students (G3 between 1 and 9 inclusive),
    sorted by G3 ascending so the worst-performing students appear first.
    Each item includes: student_index, G1, G2, G3, absences
    """

    # Filter students whose final grade falls in the "Fail" range (1-9)
    at_risk_df = df[(df["G3"] >= 1) & (df["G3"] <= 9)].copy()

    # Sort by G3 ascending (lowest/worst grades first)
    at_risk_df = at_risk_df.sort_values(by="G3", ascending=True)

    # Use the DataFrame index as the student_index identifier.
    # NOTE: the dataset has no natural/persistent student ID column, so
    # the pandas row index is used as a stand-in. This index is only
    # stable as long as `df` itself isn't re-filtered or reloaded in a
    # different row order elsewhere in the app.
    at_risk_df["student_index"] = at_risk_df.index

    # Select only the required columns, in the required order
    result = at_risk_df[["student_index", "G1", "G2", "G3", "absences"]]

    # Convert to a list of dictionaries (JSON-friendly format)
    return result.to_dict(orient="records")


# =====================================================================
# Endpoint 3: GET /top-students
# Returns the top 5 performing students by G3 (excluding dropouts)
# =====================================================================
@app.get("/top-students")
def get_top_students():
    """
    Returns the top 5 students by final grade (G3), excluding dropouts,
    sorted by G3 descending (best students first).
    Each item includes: student_index, G1, G2, G3

    NOTE: ties at the 5th-place G3 value are broken by original row
    order (pandas' default stable sort), not re-ranked by any other
    criteria. This matches "top 5" literally but isn't tie-aware.
    """

    # Exclude dropouts since G3 = 0 does not represent a real academic score
    non_dropout_df = df[df["G3"] != 0].copy()

    # Sort by G3 descending (highest grades first)
    non_dropout_df = non_dropout_df.sort_values(by="G3", ascending=False)

    # Take only the top 5 students after sorting
    top_5_df = non_dropout_df.head(5)

    # Use the DataFrame index as the student_index identifier
    top_5_df["student_index"] = top_5_df.index

    # Select only the required columns, in the required order
    result = top_5_df[["student_index", "G1", "G2", "G3"]]

    # Convert to a list of dictionaries (JSON-friendly format)
    return result.to_dict(orient="records")


# =====================================================================
# Pydantic Model: StudentInput
# Defines and validates the input data required for grade prediction
# =====================================================================
class StudentInput(BaseModel):
    """
    Input schema for the /predict-result endpoint.
    Validates ranges for each field and provides clear error messages
    if the client sends invalid data.
    """

    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="First period grade (0-20)",
        # NOTE: json_schema_extra only annotates the OpenAPI/docs schema.
        # It does NOT change the actual validation error text FastAPI
        # returns to a client on a failed request (that comes from
        # Pydantic's default messages, e.g. "Input should be <= 20").
        # A custom RequestValidationError handler would be needed for
        # that if truly custom error text is required in responses.
        json_schema_extra={"error_message": "G1 must be between 0 and 20"}
    )

    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="Second period grade (0-20)",
        json_schema_extra={"error_message": "G2 must be between 0 and 20"}
    )

    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="Weekly study time level (1-4)",
        json_schema_extra={"error_message": "studytime must be between 1 and 4"}
    )

    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of school absences (0-100)",
        json_schema_extra={"error_message": "absences must be between 0 and 100"}
    )

    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of past class failures (0-4)",
        json_schema_extra={"error_message": "failures must be between 0 and 4"}
    )


# =====================================================================
# Endpoint: POST /predict-result
# Predicts a student's final grade outcome based on early indicators
# =====================================================================
@app.post("/predict-result")
def predict_result(student: StudentInput):
    """
    Accepts early academic indicators (G1, G2, studytime, absences, failures)
    and returns a predicted final grade (estimated_g3), a categorical
    prediction (Dropout Risk / Fail / Pass), and a confidence level.
    """

    # Step 1: Calculate the estimated final grade using the given formula
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Step 2: Clamp estimated_g3 so it stays within the valid grade range (0-20)
    # NOTE: max(0, min(20, ...)) snaps exactly to the int 0 or 20 at the
    # boundaries, so the `== 0` check in Step 3 below is safe here and
    # not subject to typical floating-point equality pitfalls.
    estimated_g3 = max(0, min(20, estimated_g3))

    # Step 3: Determine the categorical prediction based on estimated_g3
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:  # estimated_g3 >= 10
        prediction = "Pass"

    # Step 4: Determine confidence level based on consistency of G1 and G2
    # Both consistently high or both consistently low -> High confidence
    # Mixed signals -> Medium confidence
    if student.G1 > 12 and student.G2 > 12:
        confidence = "High"
    elif student.G1 < 8 and student.G2 < 8:
        confidence = "High"
    else:
        confidence = "Medium"

    # Step 5: Return the results as a JSON response
    return {
        "estimated_g3": round(estimated_g3, 2),
        "prediction": prediction,
        "confidence": confidence
    }


# =====================================================================
# Root Endpoint: GET /
# Provides a basic welcome message and points users to the docs
# =====================================================================
@app.get("/")
def root():
    """
    Root endpoint that returns basic API information and a link
    to the interactive documentation.
    """
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


# =====================================================================
# MAIN BLOCK
# Runs the FastAPI app with uvicorn when this file is executed directly
# =====================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)