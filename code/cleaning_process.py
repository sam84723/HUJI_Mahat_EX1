import math
import os

import numpy as np
import pandas as pd


def normalize_country(name):
    name = name.strip()  # remove surrounding spaces
    if name.lower().startswith("the "):
        name = name[4:]  # remove "the "
    return name.title()


def clean_demographics(df, printing=False):
    cols_to_clean = ['Life Expectancy Both', 'Life Expectancy Female', 'Life Expectancy Male',
                     'UrbanPopulation Percentage', 'UrbanPopulation Absolute', 'Population Density']

    for col in cols_to_clean:
        # Convertir en numérique, les valeurs invalides deviennent NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Getting rid of the rows that contain NaN
    df = df.dropna(subset=cols_to_clean)

    # Getting rid of the negative values and the ones out of bounds (40-100)
    df = df[(df['Life Expectancy Both'] >= 40) & (df['Life Expectancy Both'] <= 100)]
    df = df[(df['Life Expectancy Female'] >= 40) & (df['Life Expectancy Female'] <= 100)]
    df = df[(df['Life Expectancy Male'] >= 40) & (df['Life Expectancy Male'] <= 100)]

    # Dealing with the name shit
    df['Original_Country'] = df['Country']

    df['Country'] = df['Country'].apply(normalize_country)

    # Find the rows where the names have changed
    mismatches = df[df['Country'] != df['Original_Country']][['Original_Country', 'Country']]
    print('Number of mismatches:', mismatches.shape[0])
    # Save into a csv
    mismatches.to_csv('../output/name_mismatches.csv', index=False)

    df = df.drop(columns=['Original_Country'])
    # df.set_index('Country', inplace=True)
    df.set_index('Country')
    return df


# a) Nettoyer et convertir GDP per capita PPP en numérique
def clean_df(value):
    if pd.isna(value):
        return None
    cleaned = ''.join(ch for ch in str(value) if ch.isdigit() or ch == '.')
    try:
        return float(cleaned)
    except ValueError:
        return None


def process_gdp_data(df_gdp, output_dir='output'):
    # Cleaning
    df_gdp['GDP_per_capita_PPP'] = df_gdp['GDP_per_capita_PPP'].apply(clean_df)

    # b) Delete the lines with NaN
    missing_gdp = df_gdp[df_gdp['GDP_per_capita_PPP'].isna()]
    if not missing_gdp.empty:
        missing_gdp.to_csv(f"{output_dir}/dropped_gdp.csv", index=False)
    df_gdp = df_gdp.dropna(subset=['GDP_per_capita_PPP'])

    # c) Identify the outliers by tukey
    Q1 = df_gdp['GDP_per_capita_PPP'].quantile(0.25)
    Q3 = df_gdp['GDP_per_capita_PPP'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_gdp[(df_gdp['GDP_per_capita_PPP'] < lower_bound) | (df_gdp['GDP_per_capita_PPP'] > upper_bound)]
    print(f"Number of GDP outliers detected : {len(outliers)}")

    df_gdp['Country'] = df_gdp['Country'].apply(normalize_country)

    # d) Check the doubles in the country column
    duplicates = df_gdp[df_gdp.duplicated(subset='Country', keep=False)]
    if not duplicates.empty:
        # Documente the doubles
        duplicates.to_csv(f"{output_dir}/duplicates_gdp.csv", index=False)
        # Only keep the first occurrence
        df_gdp = df_gdp.drop_duplicates(subset='Country', keep='first')
        # print(f"Doublons détectés et un seul enregistrement conservé par pays.")
    print('Number of duplicates:', duplicates.shape[0])

    df_gdp.set_index('Country')

    return df_gdp, outliers, missing_gdp, duplicates


def process_population_data(df_pop, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # a) Clean and convert the data
    df_pop['Population'] = df_pop['Population'].apply(clean_df)

    # b) delete the lines with missing population
    missing_pop = df_pop[df_pop['Population'].isna()]
    print(f"number of deleted lines (Missing population) : {len(missing_pop)}")
    df_pop = df_pop.dropna(subset=['Population'])

    # c) Detection of outliers (and after transformation log10)
    df_pop['Log_Population'] = df_pop['Population'].apply(lambda x: math.log10(x) if x > 0 else None)

    Q1 = df_pop['Log_Population'].quantile(0.25)
    Q3 = df_pop['Log_Population'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_pop[(df_pop['Log_Population'] < lower_bound) | (df_pop['Log_Population'] > upper_bound)]
    print(f"number of outliers in the population (log10) : {len(outliers)}")

    # d) Double verification
    duplicates = df_pop[df_pop.duplicated(subset='Country', keep=False)]
    if not duplicates.empty:
        print(f"{len(duplicates)} doublons detected.")
        duplicates.to_csv(f"{output_dir}/duplicates_population.csv", index=False)
        df_pop = df_pop.drop_duplicates(subset='Country', keep='first')
    print('Number of duplicates:', duplicates.shape[0])

    # Normalize country
    df_pop['Country'] = df_pop['Country'].apply(normalize_country)

    df_pop.set_index('Country')

    # Optionnel : supprimer la colonne temporaire log
    # df_pop = df_pop.drop(columns=['Log_Population'])

    return df_pop, outliers, missing_pop, duplicates