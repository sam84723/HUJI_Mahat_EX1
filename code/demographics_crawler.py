import requests
from bs4 import BeautifulSoup
import csv
import time
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

BASE_URL = "https://www.worldometers.info"
DEMOGRAPHICS_URL = f"{BASE_URL}/demographics/"

def get_country_links():
    """
    A function that gets the country links from the DEMOGRAPHICS website.
    :return: A list of the country links
    """
    res = requests.get(DEMOGRAPHICS_URL)
    soup = BeautifulSoup(res.content, "html.parser")
    links = []

    # Search for all the links to individual country demographics pages
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Check if the link matches the format for country demographics pages
        if href.startswith("/demographics/") and href.endswith("-demographics/"):
            country_name = a_tag.text.strip()
            full_url = BASE_URL + href
            links.append((country_name, full_url))

    return links

def find_exact_label(soup, keyword):
    """
    A function that helps find the correct text using regex
    :param soup: The BeautifulSoup object
    :param keyword: The keyword
    :return: a list of the exact label
    """
    # Search for a div whose text contains the EXACT keyword (case-insensitive)
    # \b is a "word boundary" to avoid false positives (e.g., matching "females" when looking for "male")
    pattern = re.compile(rf"\b{keyword}\b", re.IGNORECASE)
    return soup.find("div", string=lambda text: text and pattern.search(text))

def get_life_expectancy_values(soup):
    """
    A function that gets the LifeExpectancy values from a BeautifulSoup object
    :param soup: the BeautifulSoup object
    :return: the LifeExpectancy value
    """
    def find_value_by_exact_label(keyword):
        target_div = find_exact_label(soup, keyword)
        if target_div:
            prev_div = target_div.find_previous("div")
            if prev_div:
                return prev_div.text.strip()
        return ""

    val_both = find_value_by_exact_label("both sexes combined")
    val_female = find_value_by_exact_label("females")
    val_male = find_value_by_exact_label("males")

    return val_both, val_female, val_male

def extract_demographics(soup):
    """
    A function that extracts the demographic data from the soup
    :param soup:
    :return:
    """
    urban_percentage = None
    urban_absolute = None
    population_density = None

    paragraphs = soup.find_all("p")
    for p in paragraphs:
        text = p.get_text()

        # Find urban population percentage (e.g., 26.7% of the population is urban)
        match_percent = re.search(r'(\d{1,3}(?:\.\d+)?)% of the population.*urban', text)
        if match_percent:
            urban_percentage = match_percent.group(1)

        # Find urban absolute population (e.g., 11,704,638 people)
        match_absolute = re.search(r'\(([\d,]+) people in \d{4}\)', text)
        if match_absolute:
            urban_absolute = match_absolute.group(1).replace(',', '')

        # Find population density (e.g., 67 people per Km2)
        match_density = re.search(r'population density.*?is\s+([\d,]+)\s+people\s+per\s+Km2', text, re.IGNORECASE)
        if match_density:
            population_density = int(match_density.group(1).replace(",", ""))

    return urban_percentage, urban_absolute, population_density

def extract_country_data(country_name, url):
    """
    A function that extracts the country data from the url
    :param country_name: the country name
    :param url: the url of the country data
    :return: a set with the country data
    """
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    val_both, val_female, val_male = get_life_expectancy_values(soup)
    urban_percentage, urban_absolute, pop_density = extract_demographics(soup)

    return {
        "Country": country_name,
        "LifeExpectancy Both": val_both,
        "LifeExpectancy Female": val_female,
        "LifeExpectancy Male": val_male,
        "UrbanPopulation Percentage": urban_percentage,
        "UrbanPopulation Absolute": urban_absolute,
        "Population Density": pop_density
    }

# Blacklist countries/regions that we don't want to scrape
blacklist = {
    "Demographics of the Global Population",
    "Asia",
    "Africa",
    "Oceania",
    "European Union",
    "Caribbean",
    "Northern America",
    "World",
    # Add other names if necessary
}

def retrieve_data(file_name):
    """
    A function that retrieves the data from the DEMOGRAPHICS website
    :param file_name: the name of the save file
    :return:
    """
    countries = get_country_links()
    # Filter out blacklisted names
    countries = [(name, url) for name, url in countries if name not in blacklist]
    print(f"Total: {len(countries)} countries")
    print(f"{len(countries)} countries found. Starting scraping...")

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Country", "LifeExpectancy Both", "LifeExpectancy Female", "LifeExpectancy Male",
            "UrbanPopulation Percentage", "UrbanPopulation Absolute", "Population Density"
        ])
        writer.writeheader()

        for i, (country, url) in enumerate(countries):
            print(f"[{i+1}/{len(countries)}] Scraping {country}...")
            try:
                data = extract_country_data(country, url)
                writer.writerow(data)
            except Exception as e:
                print(f"Error scraping {country}: {e}")
            time.sleep(0.005)  # Sleep to avoid spamming the site

    print("Scraping finished. CSV file saved.")