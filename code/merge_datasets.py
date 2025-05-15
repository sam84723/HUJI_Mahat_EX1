import os
import numpy as np
import pandas as pd

def merge_datasets(df_demo, df_gdp, df_pop):
    # Ensure that all DataFrames have a "Country" column.
    for df, name in zip([df_demo, df_gdp, df_pop],["demographics", "GDP", "population"]):
        if 'Country' not in df.columns:
            raise KeyError(f"'Country' column not found in {name} dataset.")

    # (a) Set the "Country" column as the index in each DataFrame.
    df_demo.set_index("Country", inplace=True)
    df_gdp.set_index("Country", inplace=True)
    df_pop.set_index("Country", inplace=True)

    # Save the original sets of countries for later comparison.
    countries_demo = set(df_demo.index)
    countries_gdp = set(df_gdp.index)
    countries_pop = set(df_pop.index)
    all_countries = countries_demo.union(countries_gdp).union(countries_pop)

    # (b) Merge the three DataFrames using an inner join on the index 'Country'.
    df_merged = df_demo.join(df_gdp, how="inner").join(df_pop, how="inner")

    # (c) Record how many countries remain after the merge.
    num_remaining = df_merged.shape[0]
    print("Number of countries after inner join:", num_remaining)

    # (d) Compute and save the list of countries lost during the join.
    merged_countries = set(df_merged.index)
    lost_countries = sorted(list(all_countries - merged_countries))
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)
    lost_countries_file = os.path.join(output_dir, "lost countries.csv")
    pd.DataFrame({"Country": lost_countries}).to_csv(lost_countries_file, index=False)
    print("Lost countries saved to:", lost_countries_file)

    # (Optional) Reset the index if you prefer Country to be a regular column.
    df_merged.reset_index(inplace=True)

    # (e) Check for missing values in the merged dataset.
    missing_values = df_merged.isna().sum()
    print("Missing values per column before cleaning:\n", missing_values)

    # For numeric columns: replace missing entries with the column mean.
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df_merged[col].isnull().sum() > 0:
            df_merged[col].fillna(df_merged[col].mean(), inplace=True)

    # For categorical (non-numeric) columns: drop rows with missing values.
    categorical_cols = df_merged.select_dtypes(exclude=[np.number]).columns.tolist()
    df_merged.dropna(subset=categorical_cols, inplace=True)

    # (f) Build the final feature matrix.
    # Define the selected scaled features. Adjust this list as needed.
    selected_features = ["Life Expectancy Both", "Log_Population"]
    for col in selected_features:
        if col not in df_merged.columns:
            raise KeyError(f"Column '{col}' not found in the merged dataset.")

    # Order the merged dataset alphabetically by Country.
    df_merged.sort_values("Country", inplace=True)

    # Create the NumPy array for the selected features.
    X = df_merged[selected_features].values
    X_path = os.path.join(output_dir, "X.npy")
    np.save(X_path, X)
    print("Final feature matrix saved to:", X_path)

    # Also save the merged dataset for later use.
    merged_file = os.path.join(output_dir, "merged_data.csv")
    df_merged.to_csv(merged_file, index=False)
    print("Merged dataset saved to:", merged_file)

    return df_merged

# Example usage (if running this file directly):
if __name__ == "__main__":
    # Suppose you load your CSVs into df_demo, df_gdp, and df_pop here.
    # For instance:
    df_demo = pd.read_csv("../code/demographics_data.csv")
    df_gdp  = pd.read_csv("../code/gdp_per_capita_2021.csv", na_values=["None"])
    df_pop  = pd.read_csv("../code/population_2021.csv", na_values=["None"])
    merge_datasets(df_demo, df_gdp, df_pop)