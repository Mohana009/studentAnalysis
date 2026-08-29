import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px  # FIX: was missing — required by generate_interactive_charts()


def load_and_prepare_data(filepath):
    """
    Loads the UCI Student Performance dataset (Maths.csv) and engineers
    new columns needed for the Student Academic Risk Intelligence System.

    Parameters:
        filepath (str): Path to the Maths.csv file

    Returns:
        pd.DataFrame: The original data plus engineered columns
    """

    # Step 1: Load the CSV into a DataFrame
    df = pd.read_csv(filepath)

    # Step 2a: Classify final grade (G3) into a categorical Result
    # G3 = 0 is treated as "Dropout" (not a real academic score of zero)
    # G3 between 1-9 is a "Fail", G3 between 10-20 is a "Pass"
    def classify_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:  # 10 to 20
            return "Pass"

    df["Result"] = df["G3"].apply(classify_result)

    # Step 2b: Convert final grade to a percentage (G3 is out of 20)
    df["Percentage"] = (df["G3"] / 20) * 100

    # Step 2c: Average alcohol consumption (weekday + weekend, scaled 1-5 each)
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Step 2d: Average parental education level (Mother's + Father's)
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Step 2e: Grade trend — improvement/decline from first period to final grade
    df["grade_trend"] = df["G3"] - df["G1"]

    # Step 2f: Count how many support systems the student has
    # (school support, family support, paid classes) — count "yes" values
    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_support"] = (df[support_cols] == "yes").sum(axis=1)

    # Step 2g: Composite risk score combining multiple risk indicators
    # More past failures, more absences, more alcohol use => higher risk
    # More study time => lower risk (hence subtracted)
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Step 2h: Average of first and second period grades (G1, G2)
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Step 3: Return the enriched DataFrame
    return df


def calculate_statistics(df):
    """
    Calculates key academic risk statistics from the prepared DataFrame
    using NumPy for numerical computations.

    Parameters:
        df (pd.DataFrame): The prepared DataFrame returned by
                            load_and_prepare_data()

    Returns:
        dict: Dictionary containing total students, class average, pass rate,
              dropout count, at-risk count, and a correlation matrix of G1, G2, G3
    """

    # Step 1: Isolate non-dropout students (G3 != 0) since dropouts
    # should not distort grade-based statistics
    non_dropout_df = df[df["G3"] != 0]

    # Step 2: total_students -> simple row count of the full dataset
    # (stored directly here instead of being reconstructed later, which
    # avoids fragile algebra and edge cases like a 100% pass rate)
    total_students = len(df)

    # Step 3: class_avg_g3 -> mean final grade, excluding dropouts
    # Guard against an empty non_dropout_df (e.g. every student dropped out)
    # to avoid NaN/ZeroDivisionError from NumPy/Python.
    if len(non_dropout_df) > 0:
        # .values converts the pandas Series to a NumPy array so the
        # calculation explicitly uses NumPy (as required for this function)
        class_avg_g3 = np.mean(non_dropout_df["G3"].values)

        # Step 4: pass_rate -> % of non-dropout students who passed (G3 >= 10)
        # Calculated strictly out of non-dropout students, not the whole class
        pass_count = np.sum(non_dropout_df["G3"].values >= 10)
        pass_rate = (pass_count / len(non_dropout_df)) * 100
    else:
        class_avg_g3 = 0.0
        pass_rate = 0.0

    # Step 5: dropout_count -> total students where G3 == 0
    dropout_count = np.sum(df["G3"].values == 0)

    # Step 6: at_risk_count -> students who failed but did not drop out
    # (G3 between 1 and 9 inclusive)
    at_risk_count = np.sum((df["G3"].values >= 1) & (df["G3"].values <= 9))

    # Step 7: correlation_matrix -> correlation between G1, G2, G3
    # computed only on non-dropout students, since dropout G3=0 values
    # would artificially skew the correlation
    if len(non_dropout_df) > 1:
        grades_array = non_dropout_df[["G1", "G2", "G3"]].values
        # np.corrcoef expects variables as rows, so transpose the array
        correlation_matrix = np.corrcoef(grades_array.T)
    else:
        # Not enough data points to compute a meaningful correlation
        correlation_matrix = np.full((3, 3), np.nan)

    # Step 8: Package all results into a dictionary
    # NOTE: values are cast to native Python types (int/float) so this
    # dict is safe to use with json.dumps() or an API response later,
    # rather than leaving them as numpy.int64 / numpy.float64.
    stats = {
        "total_students": int(total_students),
        "class_avg_g3": float(class_avg_g3),
        "pass_rate": float(pass_rate),
        "dropout_count": int(dropout_count),
        "at_risk_count": int(at_risk_count),
        "correlation_matrix": correlation_matrix
    }

    return stats


def generate_static_charts(df):
    """
    Generates and saves two static charts summarizing academic performance:
    1. Bar chart of average G3 by study time level
    2. Pie chart of Pass/Fail/Dropout distribution

    Parameters:
        df (pd.DataFrame): The prepared DataFrame returned by
                            load_and_prepare_data()

    Returns:
        None (charts are saved to disk)
    """

    # Step 1: Ensure the output/ folder exists before saving anything
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Chart 1: Bar chart - Average G3 by Study Time
    # ---------------------------------------------------------------

    # Group by studytime (1,2,3,4) and compute mean G3 for each group
    avg_g3_by_studytime = df.groupby("studytime")["G3"].mean()

    # Create a new figure for the bar chart
    plt.figure(figsize=(8, 6))

    # Plot bars: x = studytime levels, y = average G3
    plt.bar(avg_g3_by_studytime.index, avg_g3_by_studytime.values, color="steelblue")

    # Add title and axis labels as specified
    plt.title("Average G3 by Study Time")
    plt.xlabel("Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Average G3")

    # Ensure all 4 studytime levels are shown as ticks on the x-axis
    plt.xticks(avg_g3_by_studytime.index)

    # Save the chart to the output folder
    plt.savefig(os.path.join(output_dir, "avg_g3_by_studytime.png"))

    # Close the figure to free memory and avoid overlapping plots
    plt.close()

    # ---------------------------------------------------------------
    # Chart 2: Pie chart - Pass / Fail / Dropout Distribution
    # ---------------------------------------------------------------

    # Count how many students fall into each Result category
    result_counts = df["Result"].value_counts()

    # Create a new figure for the pie chart
    plt.figure(figsize=(7, 7))

    # Plot pie chart with percentage labels shown on each slice
    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%",   # show percentages with 1 decimal place
        startangle=90        # start slices from the top for a cleaner look
    )

    # Add title
    plt.title("Student Result Distribution")

    # Save the chart to the output folder
    plt.savefig(os.path.join(output_dir, "pass_fail_dropout_pie.png"))

    # Close the figure to free memory and avoid overlapping plots
    plt.close()


def generate_interactive_charts(df):
    """
    Generates two interactive Plotly charts for exploring academic risk data:
    1. Scatter plot of Study Time vs Final Grade (G3), colored by Result
    2. Bar chart of Average G3 by Internet Access

    Parameters:
        df (pd.DataFrame): The prepared DataFrame returned by
                            load_and_prepare_data()

    Returns:
        None (charts are displayed via fig.show())
    """

    # ---------------------------------------------------------------
    # Chart 1: Scatter plot - Study Time vs Final Grade (G3)
    # ---------------------------------------------------------------

    # Define a fixed color mapping so results are visually consistent
    # and meaningful (green=pass, red=fail, grey=dropout)
    result_color_map = {
        "Pass": "green",
        "Fail": "red",
        "Dropout": "grey"
    }

    # Create the scatter plot using Plotly Express
    scatter_fig = px.scatter(
        df,
        x="studytime",                     # X axis: study time level
        y="G3",                            # Y axis: final grade
        color="Result",                    # Color points by Pass/Fail/Dropout
        color_discrete_map=result_color_map,  # Apply fixed color scheme
        hover_data=["absences", "G1", "G2"],  # Extra info shown on hover
        title="Study Time vs Final Grade (G3)"
    )

    # Display the scatter plot in the browser/notebook
    scatter_fig.show()

    # ---------------------------------------------------------------
    # Chart 2: Bar chart - Average G3 by Internet Access
    # ---------------------------------------------------------------

    # Group by internet access (yes/no) and compute mean G3 for each group
    avg_g3_by_internet = df.groupby("internet", as_index=False)["G3"].mean()

    # Create the bar chart using Plotly Express
    bar_fig = px.bar(
        avg_g3_by_internet,
        x="internet",                # X axis: internet access (yes/no)
        y="G3",                      # Y axis: average final grade
        color="internet",            # Color bars by internet access group
        title="Average G3 by Internet Access"
    )

    # Display the bar chart in the browser/notebook
    bar_fig.show()


def print_summary(stats):
    """
    Prints a clean, formatted summary table of the key academic
    risk statistics calculated by calculate_statistics().

    Parameters:
        stats (dict): Dictionary returned by calculate_statistics()

    Returns:
        None (prints directly to console)
    """

    # total_students now comes directly from the stats dict (calculated
    # via a simple row count in calculate_statistics), so no fragile
    # reconstruction/algebra is needed here anymore.
    print("=" * 48)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 48)
    print(f"Total Students     : {stats['total_students']}")
    print(f"Class Average G3   : {stats['class_avg_g3']:.2f}")
    print(f"Pass Rate          : {stats['pass_rate']:.2f}%")
    print(f"At-Risk Count      : {stats['at_risk_count']}")
    print(f"Dropout Count      : {stats['dropout_count']}")
    print("=" * 48)


# =====================================================================
# MAIN BLOCK
# =====================================================================
if __name__ == "__main__":

    # Step 1: Load and prepare the dataset, adding all engineered columns
    df = load_and_prepare_data("data/Maths.csv")

    # Step 2: Calculate key academic risk statistics from the prepared data
    stats = calculate_statistics(df)

    # Step 3: Generate and save static (Matplotlib) charts to output/
    generate_static_charts(df)

    # Step 4: Generate and display interactive (Plotly) charts
    generate_interactive_charts(df)

    # Step 5: Print the formatted summary table to the console
    print_summary(stats)

    # Step 6: Confirm completion to the user
    print("Analysis complete. Charts saved to output/ folder")