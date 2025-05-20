import os
import pandas as pd


def generate_feature_engineering_summary(df_merged, df_demographics):
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Updated statistics table for normalized features (after scaling)
    # Assume these are the features that were scaled
    normalized_features = ["LifeExpectancy Both", "LogGDPperCapita", "LogPopulation"]

    # Compute descriptive statistics and add median (which is not provided by describe())
    norm_stats = df_merged[normalized_features].describe().T
    norm_stats["median"] = df_merged[normalized_features].median()
    # Reorder columns to: mean, median, std, min, max
    norm_stats = norm_stats[["mean", "median", "std", "min", "max"]]

    norm_stats_file = os.path.join(output_dir, "normalized_features_desc_stats.csv")
    norm_stats.to_csv(norm_stats_file)
    print("Normalized features descriptive stats saved to:", norm_stats_file)

    # 2. Final merged dataset - number of countries and first 10 countries (alphabetically)
    # Assuming the merged dataset still contains the "Country" column after resetting the index.
    if "Country" in df_merged.columns:
        countries = sorted(df_merged["Country"].unique())
    else:
        # If the Country is the index, use:
        countries = sorted(df_merged.index.unique())

    num_countries = len(countries)
    first_10 = countries[:10]

    print(f"\nNumber of countries in the final merged dataset: {num_countries}")
    print("First 10 countries (alphabetically):")
    for c in first_10:
        print(c)

    # Save the list of first 10 countries in a CSV file for documentation.
    pd.DataFrame({"Country": first_10}).to_csv(os.path.join(output_dir, "first_10_countries.csv"), index=False)

    # 3. Overall descriptive statistics for each collected field from demographics crawling
    demographics_desc = df_demographics.describe().T
    # Add the median since .describe() doesn't include it by default
    demographics_desc["median"] = df_demographics.median(numeric_only=True)
    demographics_desc = demographics_desc[["mean", "median", "std", "min", "max"]]

    demo_desc_file = os.path.join(output_dir, "demographics_overall_desc_stats.csv")
    demographics_desc.to_csv(demo_desc_file)
    print("Demographics overall descriptive stats saved to:", demo_desc_file)

    # 4. Save a sample of your crawled data (first 5 rows)
    demo_sample_file = os.path.join(output_dir, "demographics_sample.csv")
    df_demographics.head(5).to_csv(demo_sample_file, index=False)
    print("Sample of demographics data (first 5 rows) saved to:", demo_sample_file)

    # 5. Verification of additional fields from the web crawling process.
    required_fields = [
        "LifeExpectancy Both", "LifeExpectancy Female", "LifeExpectancy Male",
        "UrbanPopulation Percentage", "UrbanPopulation Absolute", "Population Density"
    ]
    missing_fields = [field for field in required_fields if field not in df_demographics.columns]
    if not missing_fields:
        print("All additional fields collected successfully during the web crawling process.")
    else:
        print("The following required fields were missing in the demographics data:", missing_fields)

# Example usage:
# generate_feature_engineering_summary(df_merged, df_demographics)