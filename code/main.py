import os
import pandas as pd

import demographic_crawler
import cleaning_process
import feature_engineering
import merge_datasets


def data_acquisition(filename_demographics, filename_gdp, filename_pop, printing=False):
    # Ensure the output directory exists
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- DEMOGRAPHICS --------------------
    # Load the extracted demographics data into a DataFrame
    df_demographics = pd.read_csv(filename_demographics)

    # List of numeric columns to be cast. Adjust the column names if needed.
    numeric_columns = [
        "Life Expectancy (Both Sexes, in years)",
        "Life Expectancy (Females) in years",
        "Life Expectancy (Males) in years",
        "Urban Population percentage",
        "Urban Population absolute numbers",
        "Population Density per square kilometer"
    ]

    # Clean and cast numeric columns in demographics:
    for col in numeric_columns:
        if col in df_demographics.columns:
            # Remove commas (if present) and convert the column to a numeric type.
            df_demographics[col] = df_demographics[col].replace({',': ''}, regex=True)
            df_demographics[col] = pd.to_numeric(df_demographics[col], errors='coerce')

    # Save the cleaned demographics DataFrame to output/demographics_data.csv
    demographics_data_path = os.path.join(output_dir, "demographics_data.csv")
    df_demographics.to_csv(demographics_data_path, index=False)
    if printing:
        print(f"Cleaned demographics DataFrame saved to {demographics_data_path}\n")

    # (d) Print the first 10 rows BEFORE sorting demographics
    before_sort = df_demographics.head(10)
    if printing:
        print("---- First 10 rows of demographics BEFORE sorting ----")
        print(before_sort)
    before_sort_path = os.path.join(output_dir, "demographics_before_sort.csv")
    before_sort.to_csv(before_sort_path, index=False)
    if printing:
        print(f"\nFirst 10 rows before sort saved to {before_sort_path}\n")

    # (d) Sort the demographics DataFrame by the 'Country' column (if available)
    if "Country" in df_demographics.columns:
        df_sorted = df_demographics.sort_values(by="Country", ascending=True)
    else:
        if printing:
            print("Warning: 'Country' column not found in demographics. Skipping sort by Country.")
        df_sorted = df_demographics

    # Print the first 10 rows AFTER sorting demographics
    after_sort = df_sorted.head(10)
    if printing:
        print("---- First 10 rows of demographics AFTER sorting by 'Country' ----")
        print(after_sort)
    after_sort_path = os.path.join(output_dir, "demographics_after_sort.csv")
    after_sort.to_csv(after_sort_path, index=False)
    if printing:
        print(f"\nFirst 10 rows after sort saved to {after_sort_path}\n")

    # -------------------- GDP & POPULATION --------------------
    # Read the GDP and Population CSV files into DataFrames with "None" interpreted as a missing value.
    df_gdp = pd.read_csv(filename_gdp, na_values="None")
    df_pop = pd.read_csv(filename_pop, na_values="None")

    if printing:
        print("GDP DataFrame (unsorted):")
        print(df_gdp.head())
        print("\nPopulation DataFrame (unsorted):")
        print(df_pop.head())

    # (c) Ensure numeric types for the GDP and Population columns.
    # Adjust the column names if needed.
    if "GDP" in df_gdp.columns:
        df_gdp["GDP"] = pd.to_numeric(df_gdp["GDP"], errors="coerce")
    else:
        if printing:
            print("Warning: 'GDP' column not found in df_gdp.")

    if "Population" in df_pop.columns:
        df_pop["Population"] = pd.to_numeric(df_pop["Population"], errors="coerce")
    else:
        if printing:
            print("Warning: 'Population' column not found in df_pop.")

    # (d) Print and save the first 5 rows BEFORE sorting GDP and Population DataFrames
    gdp_before_sort = df_gdp.head(5)
    gdp_before_sort_path = os.path.join(output_dir, "gdp before sort.csv")
    gdp_before_sort.to_csv(gdp_before_sort_path, index=False)
    if printing:
        print("\nGDP DataFrame - BEFORE sorting (first 5 rows):")
        print(gdp_before_sort)
        print(f"Saved to {gdp_before_sort_path}")

    pop_before_sort = df_pop.head(5)
    pop_before_sort_path = os.path.join(output_dir, "pop before sort.csv")
    pop_before_sort.to_csv(pop_before_sort_path, index=False)
    if printing:
        print("\nPopulation DataFrame - BEFORE sorting (first 5 rows):")
        print(pop_before_sort)
        print(f"Saved to {pop_before_sort_path}")

    # Sorting by the 'Country' column if it exists
    if "Country" in df_gdp.columns:
        df_gdp_sorted = df_gdp.sort_values(by="Country", ascending=True)
    else:
        if printing:
            print("Warning: 'Country' column not found in df_gdp; skipping sort.")
        df_gdp_sorted = df_gdp

    if "Country" in df_pop.columns:
        df_pop_sorted = df_pop.sort_values(by="Country", ascending=True)
    else:
        if printing:
            print("Warning: 'Country' column not found in df_pop; skipping sort.")
        df_pop_sorted = df_pop

    # Print and save the first 5 rows AFTER sorting GDP and Population DataFrames
    gdp_after_sort = df_gdp_sorted.head(5)
    gdp_after_sort_path = os.path.join(output_dir, "gdp after sort.csv")
    gdp_after_sort.to_csv(gdp_after_sort_path, index=False)
    if printing:
        print("\nGDP DataFrame - AFTER sorting by 'Country' (first 5 rows):")
        print(gdp_after_sort)
        print(f"Saved to {gdp_after_sort_path}")

    pop_after_sort = df_pop_sorted.head(5)
    pop_after_sort_path = os.path.join(output_dir, "pop after sort.csv")
    pop_after_sort.to_csv(pop_after_sort_path, index=False)
    if printing:
        print("\nPopulation DataFrame - AFTER sorting by 'Country' (first 5 rows):")
        print(pop_after_sort)
        print(f"Saved to {pop_after_sort_path}")

    # (e) Run describe() on both DataFrames and save the resulting tables.
    gdp_describe = df_gdp.describe()
    gdp_describe_path = os.path.join(output_dir, "gdp describe.csv")
    gdp_describe.to_csv(gdp_describe_path)
    if printing:
        print("\nGDP DataFrame - Describe():")
        print(gdp_describe)
        print(f"Saved to {gdp_describe_path}")

    pop_describe = df_pop.describe()
    pop_describe_path = os.path.join(output_dir, "pop describe.csv")
    pop_describe.to_csv(pop_describe_path)
    if printing:
        print("\nPopulation DataFrame - Describe():")
        print(pop_describe)
        print(f"Saved to {pop_describe_path}")

    return df_demographics, df_gdp, df_pop



def main():
    file_name_demo = "code/demographics_data.csv"
    gdp_file = "code/gdp_per_capita_2021.csv"
    pop_file = "code/population_2021.csv"

    # demographic_crawler.retrieve_data(file_name_demo)

    # Crawling our way to the data
    if not os.path.exists(file_name_demo):
        demographic_crawler.retrieve_data(file_name_demo)
    else:
        print("File already exists. Skipping the crawling.")

    df_demographics, df_gdp, df_pop =  data_acquisition(file_name_demo, gdp_file, pop_file, printing=True)


    df_demographics = cleaning_process.clean_demographics(df_demographics)
    gdp_results = cleaning_process.process_gdp_data(df_gdp)
    pop_results = cleaning_process.process_population_data(df_pop)

    df_merged = merge_datasets.merge_datasets(df_demographics, gdp_results[0], pop_results[0])
    feature_engineering.feature_engineering(df_merged)

if __name__ == "__main__":
    main() 