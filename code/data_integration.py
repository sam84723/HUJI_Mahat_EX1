import os
import numpy as np
import pandas as pd



def main():
    print("Current working directory:", os.getcwd())
    demographics_file = "./demographics_data.csv"  # cleaned demographics data
    gdp_file = "../code/gdp_per_capita_2021.csv"  # cleaned GDP data
    pop_file = "../code/population_2021.csv"  # cleaned Population data
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------- Country Name Canonicalization ---------------------
    # Define a dictionary mapping variants to the canonical country name.
    # You can manually update this dictionary as needed.
    country_mapping = {
        "Cape Verde": "Cabo Verde",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "DR Congo": "Democratic Republic of Congo",
        "Faeroe Islands": "Faroe Islands",
        "Micronesia (country)": "Micronesia",
        "Réunion": "Reunion",
        "Sao Tome & Principe": "Sao Tome and Principe",
        # Add other mappings here as necessary.
    }

    # Function to return canonical name for a given country.
    def clean_country(name):
        # Remove any extra whitespace and convert to a consistent format.
        name = name.strip()
        return country_mapping.get(name, name)

    # ----------------------------- Data Loading -----------------------------
    # (a) Load the three DataFrames.
    df_demographics = pd.read_csv(demographics_file)
    df_gdp = pd.read_csv(gdp_file)
    df_pop = pd.read_csv(pop_file)

    # -------------------- Apply Country Name Canonicalization --------------------
    # Clean country names in each dataset by applying the mapping function.
    if "Country" not in df_demographics.columns or "Country" not in df_gdp.columns or "Country" not in df_pop.columns:
        raise KeyError("One of the datasets does not have a 'Country' column.")

    df_demographics["Country"] = df_demographics["Country"].apply(clean_country)
    df_gdp["Country"] = df_gdp["Country"].apply(clean_country)
    df_pop["Country"] = df_pop["Country"].apply(clean_country)

    # ------------------------- Set Country as Index -------------------------
    df_demographics.set_index("Country", inplace=True)
    df_gdp.set_index("Country", inplace=True)
    df_pop.set_index("Country", inplace=True)

    # ------------------ Conjoin Duplicate Countries (if any) ------------------
    # If a country appears more than once due to variations, group by the index to consolidate.
    df_demographics = df_demographics.groupby(level=0).first()
    df_gdp = df_gdp.groupby(level=0).first()
    df_pop = df_pop.groupby(level=0).first()

    # ----------------- Record Country Sets for Lost Countries ----------------
    countries_demographics = set(df_demographics.index)
    countries_gdp = set(df_gdp.index)
    countries_pop = set(df_pop.index)
    all_countries = countries_demographics.union(countries_gdp).union(countries_pop)

    # ------------------- 5.4 Data Integration -------------------
    # (b) Perform an inner join across all DataFrames on Country.
    df_final = df_demographics.join(df_gdp, how="inner").join(df_pop, how="inner")

    # (c) Record how many countries remain after the merge.
    remaining_countries = df_final.shape[0]
    print(f"Number of countries after the inner join: {remaining_countries}")

    # (d) Save the list of countries lost during the join.
    # Lost countries are those in the union but not in the final joined DataFrame.
    lost_countries = sorted(list(all_countries - set(df_final.index)))
    df_lost = pd.DataFrame({"Country": lost_countries})
    lost_countries_path = os.path.join(output_dir, "lost_countries.csv")
    df_lost.to_csv(lost_countries_path, index=False)
    print(f"List of lost countries saved to {lost_countries_path}")

    # (e) Check for missing values in df_final and fix them:
    # Identify numeric and categorical columns.
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_final.select_dtypes(exclude=[np.number]).columns.tolist()

    # For numeric columns, replace missing entries with the column mean.
    for col in numeric_cols:
        if df_final[col].isnull().any():
            mean_val = df_final[col].mean()
            df_final[col].fillna(mean_val, inplace=True)

    # For categorical columns, remove rows with missing values.
    if categorical_cols:
        df_final.dropna(subset=categorical_cols, inplace=True)

    # (f) Build the final feature matrix.
    # We assume the final merged dataset contains the following (scaled or to-be-scaled) columns:
    #   "LifeExpectancy Both", "LogGDPperCapita", "LogPopulation"
    # If they are not already scaled, we apply z-score normalization here.

    scaled_features = ["LifeExpectancy Both", "LogGDPperCapita", "LogPopulation"]
    for col in scaled_features:
        if col not in df_final.columns:
            raise KeyError(f"Column '{col}' not found in the merged dataset.")
        # Compute z-score normalization using the cleaned values.
        mu = df_final[col].mean()
        sigma = df_final[col].std(ddof=0)
        df_final[col] = (df_final[col] - mu) / sigma

    # Order the dataset alphabetically by Country (i.e. by index).
    df_final.sort_index(inplace=True)

    # Create the final feature matrix as a NumPy array.
    X = df_final.loc[:, scaled_features].to_numpy()
    X_path = os.path.join(output_dir, "X.npy")
    np.save(X_path, X)
    print(f"Final feature matrix saved to {X_path}")


if __name__ == "__main__":
    main()