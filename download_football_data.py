"""
Script to download football data from football-data.co.uk
Downloads historical data from 2014-2019 to extend existing dataset
"""

import os
import time

import requests

# Base URL for football-data.co.uk
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# League codes:
# E0 = English Premier League
# E1 = English Championship
# D1 = German Bundesliga 1
# D2 = German Bundesliga 2
# F1 = French Ligue 1
# F2 = French Ligue 2
# I1 = Italian Serie A
# I2 = Italian Serie B
# SP1 = Spanish La Liga
# SP2 = Spanish Segunda Division

LEAGUES = ["E0", "E1", "D1", "D2", "F1", "F2", "I1", "I2", "SP1", "SP2"]

# Seasons to download (extending backwards from 2019)
SEASONS = {
    "1819": "2018-19",
    "1718": "2017-18",
    "1617": "2016-17",
    "1516": "2015-16",
    "1415": "2014-15",
}


def download_data(
    season_code, season_name, leagues, base_dir="Football Machine/data/raw"
):
    """
    Download football data for a specific season

    Args:
        season_code: Short code like '1819' for 2018-19
        season_name: Full name like '2018-19'
        leagues: List of league codes to download
        base_dir: Base directory to save data
    """
    # Create season directory
    season_dir = os.path.join(base_dir, season_code)
    os.makedirs(season_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Downloading data for {season_name} season")
    print(f"{'=' * 60}")

    success_count = 0
    failed_leagues = []

    for league in leagues:
        # Construct URL
        url = f"{BASE_URL}/{season_code}/{league}.csv"
        file_path = os.path.join(season_dir, f"{league}.csv")

        try:
            print(f"Downloading {league} ({season_name})... ", end="")
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Save the file
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"✓ Success ({len(response.content)} bytes)")
                success_count += 1
            else:
                print(f"✗ Failed (HTTP {response.status_code})")
                failed_leagues.append(league)

            # Be nice to the server
            time.sleep(0.5)

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            failed_leagues.append(league)

    print(f"\n{'-' * 60}")
    print(
        f"Season {season_name}: {success_count}/{len(leagues)} leagues downloaded successfully"
    )
    if failed_leagues:
        print(f"Failed leagues: {', '.join(failed_leagues)}")
    print(f"{'-' * 60}")

    return success_count, failed_leagues


def main():
    print("\n" + "=" * 60)
    print("Football Data Downloader - football-data.co.uk")
    print("Downloading historical data: 2014-15 to 2018-19")
    print("=" * 60)

    total_success = 0
    total_files = 0
    all_failed = {}

    for season_code, season_name in SEASONS.items():
        success, failed = download_data(season_code, season_name, LEAGUES)
        total_success += success
        total_files += len(LEAGUES)
        if failed:
            all_failed[season_name] = failed

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files downloaded: {total_success}/{total_files}")

    if all_failed:
        print("\nFailed downloads:")
        for season, leagues in all_failed.items():
            print(f"  {season}: {', '.join(leagues)}")
    else:
        print("\n✓ All files downloaded successfully!")

    print("\nData saved to: Football Machine/data/raw/")
    print("Seasons downloaded: 1415, 1516, 1617, 1718, 1819")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
