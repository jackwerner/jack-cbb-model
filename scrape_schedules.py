import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pandas as pd

def get_scores_for_date(date_str):
    """
    Scrape scores for a specific date from Sports Reference
    """
    # Convert date string to datetime and format it correctly for the URL
    date = datetime.strptime(date_str, "%Y-%m-%d")
    url = f"https://www.sports-reference.com/cbb/boxscores/index.cgi?month={date.month}&day={date.day}&year={date.year}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        # Find all game rows
        for game in soup.find_all('tbody'):
            rows = game.find_all('tr')
            for i in range(0, len(rows), 2):
                if i + 1 < len(rows):
                    team1_row = rows[i]
                    team2_row = rows[i + 1]
                    
                    # Extract game info
                    try:
                        team1 = team1_row.find('a').text
                        team2 = team2_row.find('a').text
                        score1 = team1_row.find_all('td')[1].text
                        score2 = team2_row.find_all('td')[1].text
                        
                        # Get tournament/conference info if available
                        tournament_info = team2_row.find_next_sibling('tr')
                        tournament = tournament_info.text.strip() if tournament_info else "Regular Season"
                        
                        games.append({
                            'date': date_str,
                            'team1': team1,
                            'team2': team2,
                            'score1': score1,
                            'score2': score2,
                            'tournament': tournament
                        })
                    except (AttributeError, IndexError):
                        continue
                        
        return games
    
    except requests.RequestException as e:
        print(f"Error fetching data for {date_str}: {e}")
        return []

def main():
    # Set date range for 2025 season (typically starts in November of previous year)
    start_date = datetime(2024, 11, 1)
    end_date = datetime.now()
    
    all_games = []
    current_date = start_date
    days_counter = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Scraping games for {date_str}")
        
        games = get_scores_for_date(date_str)
        all_games.extend(games)
        
        days_counter += 1
        # Save checkpoint every 10 days
        if days_counter % 10 == 0:
            checkpoint_df = pd.DataFrame(all_games)
            checkpoint_file = f'basketball_scores_2025_checkpoint_{date_str}.csv'
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"Saved checkpoint with {len(checkpoint_df)} games to {checkpoint_file}")
        
        # Add delay to be respectful to the server
        time.sleep(3)
        current_date += timedelta(days=1)
    
    # Save final results
    df = pd.DataFrame(all_games)
    df.to_csv('basketball_scores_2025.csv', index=False)
    print(f"Saved {len(df)} games to basketball_scores_2025.csv")

if __name__ == "__main__":
    main()
