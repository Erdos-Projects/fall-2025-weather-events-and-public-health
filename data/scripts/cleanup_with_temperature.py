import pandas as pd


# Read a file and clean it up
def cleanup_csv(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.rename(columns={"End Year": "Year"})

    # Keep only the most recent year if there are multiple
    df = df[df["Year"] == df["Year"].max()]
    df = df[["County", "Value"]]

    return df


# Merge with good names for the columns
def merge_csv(df: pd.DataFrame, file: str, name: str):
    df = pd.merge(df, cleanup_csv(file), how="inner", on=["County"], suffixes=(None, None))
    df = df.rename(columns={"Value": name})

    return df


def merge_temp_month(left_df, csv_path, month, left_on="County", right_on="County"):
    right = pd.read_csv(csv_path, usecols=[right_on, month])
    right[right_on] = right[right_on].astype(str)
    left = left_df.copy()
    left[left_on] = left[left_on].astype(str)

    left = left.merge(right, how="left", left_on=left_on, right_on=right_on)
    return left


# Slightly different work for the Excel sheets
def cleanup_xlsx(file: str) -> pd.DataFrame:
    df = pd.read_excel(file)

    # Drop California aggregate row
    # New way of doing it fixes issue that dropped Alameda
    if 'County' in df.columns:
        df = df[df['County'] == 'California']

    df = df[["Counties", "Age-adjusted rate per 100,000"]]
    df = df.rename(columns={"Counties": "County"})

    return df


def merge_xlsx(df: pd.DataFrame, file: str, name: str):
    df = pd.merge(df, cleanup_xlsx(file), how="left", on=["County"], suffixes=(None, None))
    df = df.rename(columns={"Age-adjusted rate per 100,000": name})

    return df


# Read a file and clean it up
# Link to NOAA data https://www.ncei.noaa.gov/pub/data/cirs/climdiv/

# returns df, all data in the file
# also returns cal, only California data
def cleanup_NOAA_txt(file: str) -> pd.DataFrame:
    col_names = ["County Data", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # County Data interpretation: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/county-readme.txt
    df = pd.read_csv(file, delim_whitespace=True, names=col_names, dtype={"County Data": str})
    df["State Code"] = df["County Data"].str[:2]
    df["FIPS"] = df["County Data"].str[2:5]
    df["Year"] = df["County Data"].str[-4:]

    # State code "04" is California
    cal = df[df["State Code"] == "04"]
    cal = cal.drop(columns=['County Data', 'State Code'])

    return df, cal


# Map of all california counties and their FIPS codes
CA_COUNTY_FIPS = {
    "001": "Alameda", "003": "Alpine", "005": "Amador", "007": "Butte", "009": "Calaveras",
    "011": "Colusa", "013": "Contra Costa", "015": "Del Norte", "017": "El Dorado", "019": "Fresno",
    "021": "Glenn", "023": "Humboldt", "025": "Imperial", "027": "Inyo", "029": "Kern",
    "031": "Kings", "033": "Lake", "035": "Lassen", "037": "Los Angeles", "039": "Madera",
    "041": "Marin", "043": "Mariposa", "045": "Mendocino", "047": "Merced", "049": "Modoc",
    "051": "Mono", "053": "Monterey", "055": "Napa", "057": "Nevada", "059": "Orange",
    "061": "Placer", "063": "Plumas", "065": "Riverside", "067": "Sacramento", "069": "San Benito",
    "071": "San Bernardino", "073": "San Diego", "075": "San Francisco", "077": "San Joaquin",
    "079": "San Luis Obispo", "081": "San Mateo", "083": "Santa Barbara", "085": "Santa Clara",
    "087": "Santa Cruz", "089": "Shasta", "091": "Sierra", "093": "Siskiyou", "095": "Solano",
    "097": "Sonoma", "099": "Stanislaus", "101": "Sutter", "103": "Tehama", "105": "Trinity",
    "107": "Tulare", "109": "Tuolumne", "111": "Ventura", "113": "Yolo", "115": "Yuba",
}


# takes in dataframe with FIPS
# If column "County" exists, returns df
# Else, adds new column with county name
def addCountyName(df: pd.DataFrame) -> pd.DataFrame:
    if "County" in df.columns:
        return df
    s = df["FIPS"].str.strip().str.zfill(3)
    df["County"] = s.map(CA_COUNTY_FIPS)
    return df


# We'll join all the county stats together

# Start with the Excel sheets
county_stats = cleanup_xlsx("../Emergency Department_Visits_Age-adjusted_rate_per_100000_2023_Counties.xlsx")
county_stats = county_stats.rename(columns={"Age-adjusted rate per 100,000": "Emergency Visits / 100000"})

county_stats = merge_xlsx(county_stats, "../Hospitalizations_Age-adjusted_rate_per_100000_2023_Counties.xlsx",
                          "Hospitalizations / 100000")

# Creates max temp and Cooling Degree Days (CDD) dataframes for each year (1895-2015)
maxTempUS, maxTempCA = cleanup_NOAA_txt("../climdiv-tmaxcy-v1.0.0-20250905.txt")
cddUS, cddCA = cleanup_NOAA_txt("../climdiv-cddccy-v1.0.0-20250905.txt")

# Chose to drop months, based on a CDC manuscript doing the same when studying heat related illness
# Vaidyanathan A, Gates A, Brown C, Prezzato E, Bernstein A. Heat-Related Emergency Department Visits
# — United States, May–September 2023. MMWR Morb Mortal Wkly Rep 2024;73:324–329.
# DOI: http://dx.doi.org/10.15585/mmwr.mm7315a1
months_to_drop = ["Jan", "Feb", "Mar", "Apr", "Oct", "Nov", "Dec"]
maxTempSuCA = maxTempCA.drop(columns=months_to_drop)
cddSuCA = cddCA.drop(columns=months_to_drop)

# Dataset for 2023 CA max Temp and CDD
# Align with 2023 heat illness data we have
maxTempSu2023CA = maxTempSuCA[maxTempCA["Year"] == "2023"]
maxTempSu2023CA = maxTempSu2023CA.drop(columns=['Year'])
cddSu2023CA = cddSuCA[cddSuCA["Year"] == "2023"]
cddSu2023CA = cddSu2023CA.drop(columns=['Year'])

# Adds county names
maxTempSu2023CACounty = addCountyName(maxTempSu2023CA)
cddSu2023CACounty = addCountyName(cddSu2023CA)
maxTempSu2023CACounty.to_csv('../maxTempSu2023CACounty.csv', index=False)
cddSu2023CACounty.to_csv('../cddSu2023CACounty.csv', index=False)

# Join the CSVs
county_stats = merge_csv(county_stats, "../Avg_annual_energy_burden_percent_of_income_2018.csv",
                         "Energy Burden % of Income")
county_stats = merge_csv(county_stats, "../Avg_percent_of_imperviousness_2021.csv", "Imperviousness")
county_stats = merge_csv(county_stats, "../Distance_to_parks_half-mile_2010_2015_2020.csv", "Park within 1/2 Mile")
county_stats = merge_csv(county_stats, "../Hospital_beds_per_10000_population_2020.csv", "Hospital Beds / 10000")
county_stats = merge_csv(county_stats, "../Housing_built_before_1980.csv", "Housing Built before 1980")
county_stats = merge_csv(county_stats, "../Housing_insecurity_2022.csv", "Housing Insecurity")
county_stats = merge_csv(county_stats, "../Lack_of_reliable_transportation_2022.csv", "Lack of Reliable Transportation")
county_stats = merge_csv(county_stats, "../Percent_without_internet_2018-2022.csv", "% w/o Internet")
county_stats = merge_csv(county_stats, "../Utility_services_threat_2022.csv", "Utility Services Threat")
county_stats = merge_temp_month(county_stats, "../maxTempSu2023CACounty.csv", "Jul")
county_stats = county_stats.rename(columns={"Jul": "July max temp (F)"})
county_stats = merge_temp_month(county_stats, "../maxTempSu2023CACounty.csv", "Aug")
county_stats = county_stats.rename(columns={"Aug": "August max temp (F)"})
county_stats = merge_temp_month(county_stats, "../cddSu2023CACounty.csv", "Jul")
county_stats = county_stats.rename(columns={"Jul": "July CDD"})
county_stats = merge_temp_month(county_stats, "../cddSu2023CACounty.csv", "Aug")
county_stats = county_stats.rename(columns={"Aug": "August CDD"})

# Save it
county_stats.to_excel("../County_Statistics_withTemp.xlsx", index=False)

# Now we will filter the Master CVI dataset
CVI_df = pd.read_excel("../Master CVI Dataset - Oct 2023.xlsx")
CVI_df = CVI_df[CVI_df.State == "CA"]

# And attach the Low Food Access data, since it's also organized by census tract
low_food_access_df = pd.read_csv("../Low_income_Low_Food_Access_by_Census_Tracts_2019_2015.csv")
low_food_access_df = low_food_access_df[low_food_access_df.Year == 2019]
low_food_access_df = low_food_access_df[["CensusTract", "Food Access"]]

CVI_df = pd.merge(CVI_df, low_food_access_df, how="left", left_on=["FIPS Code"], right_on=["CensusTract"])
CVI_df = CVI_df.drop(["State", "CensusTract"], axis=1)

CVI_df.to_excel("../California_CVI_dataset.xlsx", index=False)
