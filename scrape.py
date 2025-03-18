import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_table(url):
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Send request to the website
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        
        if table:
            # Extract headers
            headers = []
            header_row = table.find('thead').find_all('tr')[-1]  # Get the last row in thead
            for th in header_row.find_all('th'):
                headers.append(th.text.strip())
            
            # Extract data rows
            rows = []
            for tr in table.find('tbody').find_all('tr'):
                # Skip rows that are actually duplicate headers
                if 'thead' in tr.get('class', []) or tr.find('th', scope='col'):
                    continue
                
                row = []
                for td in tr.find_all(['td', 'th']):
                    row.append(td.text.strip())
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            return df
        else:
            print(f"No table found on {url}")
            return None
    else:
        print(f"Failed to retrieve the page: {response.status_code}")
        return None

def get_combined_stats():
    # URLs to scrape
    urls = [
        'https://www.sports-reference.com/cbb/seasons/men/2025-school-stats.html',
        'https://www.sports-reference.com/cbb/seasons/men/2025-opponent-stats.html',
        'https://www.sports-reference.com/cbb/seasons/men/2025-advanced-school-stats.html',
        'https://www.sports-reference.com/cbb/seasons/men/2025-advanced-opponent-stats.html'
    ]
    
    # Prefixes for column names to distinguish between different tables
    prefixes = [
        'team_',
        'opp_',
        'adv_team_',
        'adv_opp_'
    ]
    
    dataframes = []
    
    # Scrape each URL
    for i, url in enumerate(urls):
        print(f"Scraping {url}...")
        df = scrape_table(url)
        
        if df is not None:
            # More robust duplicate header removal
            if 'School' in df.columns:
                # Remove any rows where School column contains 'School' or 'Overall'
                df = df[~df['School'].isin(['School', 'Overall', 'Conf.', 'Home', 'Away', 'Points'])]
            
            # Rename columns to avoid duplicates, except for 'School'
            prefix = prefixes[i]
            df.columns = [col if col == 'School' else f"{prefix}{col}" for col in df.columns]
            
            # Add to list of dataframes
            dataframes.append(df)
        
        # Add a delay between requests to avoid being blocked
        if i < len(urls) - 1:
            print("Waiting before next request...")
            time.sleep(3)
    
    # Join all dataframes on 'School' column
    if dataframes:
        combined_df = dataframes[0]
        for df in dataframes[1:]:
            combined_df = pd.merge(combined_df, df, on='School', how='outer')
        
        return combined_df
    else:
        print("No data was scraped.")
        return None

if __name__ == "__main__":
    # When run as a script, get the combined stats and save to CSV
    combined_stats = get_combined_stats()
    
    if combined_stats is not None:
        combined_stats.to_csv('combined_cbb_stats_2025.csv', index=False)
        print("Combined stats saved to combined_cbb_stats_2025.csv")
        print(f"Total schools: {len(combined_stats)}")
        print(f"Total columns: {len(combined_stats.columns)}")
    
    print("Scraping completed!")
