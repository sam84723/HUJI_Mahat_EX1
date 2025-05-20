import os
import numpy as np
import pandas as pd


def feature_engineering(df):
    output_dir = "../output"
    # ---------------------- 5.1 New Feature: Total GDP ----------------------
    # Ensure the required columns exist
    required_cols = ["GDP_per_capita_PPP", "Population"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the merged dataset. Please check the column names.")

    # Convert these columns to numeric in case they're not already
    df["GDP_per_capita_PPP"] = pd.to_numeric(df["GDP_per_capita_PPP"], errors="coerce")
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")

    # Ensure that all values for GDP per capita and Population are positive for both multiplication and log transforms
    if (df["GDP_per_capita_PPP"] <= 0).any():
        raise ValueError(
            "All values in 'GDP_per_capita_PPP' must be positive for correct log transformation and calculations.")
    if (df["Population"] <= 0).any():
        raise ValueError("All values in 'Population' must be positive for correct log transformation and calculations.")

    # Create the Total GDP feature
    df["TotalGDP"] = df["GDP_per_capita_PPP"] * df["Population"]

    # ---------------------- 5.2 Log Transformations ----------------------
    # Compute log10 transformation for GDP per capita PPP and Population.
    # (LifeExpectancy Both is not transformed.)
    df["LogGDPperCapita"] = np.log10(df["GDP_per_capita_PPP"])
    df["LogPopulation"] = np.log10(df["Population"])

    # ---------------------- 5.3 Scaling (Z-score Normalization) ----------------------
    # Define the three columns to normalize.
    features_to_normalize = ["LifeExpectancy Both", "LogGDPperCapita", "LogPopulation"]
    # Check that these columns exist.
    for col in features_to_normalize:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the merged dataset. Please verify your data.")

    # Compute z-score normalization for each feature.
    normalized_features = {}
    for col in features_to_normalize:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)  # using population standard deviation; adjust ddof if needed.
        normalized_features[col] = (df[col] - mu) / sigma

    # Create the final feature matrix from the normalized columns.
    feature_matrix = pd.DataFrame(normalized_features)

    # Save the feature matrix as a numpy array in output/X.npy
    feature_matrix_path = os.path.join(output_dir, "X.npy")
    np.save(feature_matrix_path, feature_matrix.values)
    print("Feature matrix (normalized) saved to:", feature_matrix_path)

    # Optionally, save the updated merged dataset (including the new features) for reference.
    merged_output_file = os.path.join(output_dir, "merged_data_with_features.csv")
    df.to_csv(merged_output_file, index=False)
    print("Updated merged dataset with the new features saved to:", merged_output_file)


if __name__ == "__main__":
    main()