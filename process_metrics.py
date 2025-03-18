import pandas as pd
import scrape

def calculate_metrics():
    # Option 1: Use the scrape module to get fresh data
    # combined_stats = scrape.get_combined_stats()
    
    # Option 2: Load the existing CSV file
    combined_stats = pd.read_csv('combined_cbb_stats_2025.csv', low_memory=False)
    
    # Convert string columns to numeric
    numeric_columns = ['opp_TOV', 'team_TOV', 'team_TRB', 'opp_TRB', 'team_3PA', 'team_3P%', 
                       'team_FGA', 'team_FG', 'team_3P', 'team_FTA', 'team_FT%', 'team_G',
                       'adv_team_Pace', 'adv_opp_Pace']
    
    for col in numeric_columns:
        combined_stats[col] = pd.to_numeric(combined_stats[col], errors='coerce')
    
    # Normalize counting stats by games played (team_G)
    counting_stats = ['opp_TOV', 'team_TOV', 'team_TRB', 'opp_TRB', 'team_3PA', 
                      'team_FGA', 'team_FG', 'team_3P', 'team_FTA']
    
    for col in counting_stats:
        combined_stats[col] = combined_stats[col] / combined_stats['team_G']
    
    # Calculate turnover margin (team turnovers - opponent turnovers)
    combined_stats['turnover_margin'] = combined_stats['opp_TOV'] - combined_stats['team_TOV']
    
    # Calculate rebound margin (team rebounds - opponent rebounds)
    combined_stats['rebound_margin'] = combined_stats['team_TRB'] - combined_stats['opp_TRB']
    
    # Calculate points from different shot types
    
    # Points from 3-pointers (3PA * 3P% * 3)
    combined_stats['points_from_3pt'] = combined_stats['team_3PA'] * combined_stats['team_3P%'] * 3
    
    # Points from 2-pointers
    # First calculate 2-point attempts (FGA - 3PA)
    combined_stats['team_2PA'] = combined_stats['team_FGA'] - combined_stats['team_3PA']
    
    # Calculate 2-point percentage
    # (FG - 3P) / 2PA
    combined_stats['team_2P'] = combined_stats['team_FG'] - combined_stats['team_3P']
    combined_stats['team_2P%'] = combined_stats['team_2P'] / combined_stats['team_2PA']
    
    # Points from 2-pointers (2PA * 2P% * 2)
    combined_stats['points_from_2pt'] = combined_stats['team_2PA'] * combined_stats['team_2P%'] * 2
    
    # Points from free throws (FTA * FT%)
    combined_stats['points_from_ft'] = combined_stats['team_FTA'] * combined_stats['team_FT%']
    
    # Total points
    combined_stats['calculated_total_points'] = (
        combined_stats['points_from_3pt'] + 
        combined_stats['points_from_2pt'] + 
        combined_stats['points_from_ft']
    )
    
    # Summarize turnovers and rebounds into 'posession_margin' column
    combined_stats['posession_margin'] = combined_stats['team_TOV'] + combined_stats['team_TRB']

    # Create a results dataframe with the key metrics
    results = combined_stats[['School', 'team_SOS', 'team_2PA', 'team_2P%', 'team_3PA', 'team_3P%',
                             'team_FTA', 'team_FT%', 'turnover_margin', 'rebound_margin', 
                             'adv_team_Pace']].copy()
    
    # Sort by total points descending (can modify this if you want to sort by a different metric)
    results = results.sort_values('team_2PA', ascending=False)
    
    # Save results to CSV
    results.to_csv('team_metrics_analysis.csv', index=False)
    
    return results

if __name__ == "__main__":
    print("Calculating team metrics...")
    results = calculate_metrics()
    
    # Get team names from user
    team1 = 'Duke'
    team2 = 'Florida'
    
    # Find teams in results (case-insensitive)
    team1_data = results[results['School'].str.lower() == team1.lower()]
    team2_data = results[results['School'].str.lower() == team2.lower()]
    
    if team1_data.empty:
        print(f"Team '{team1}' not found. Please check the spelling.")
    if team2_data.empty:
        print(f"Team '{team2}' not found. Please check the spelling.")
    
    if not team1_data.empty and not team2_data.empty:
        print(f"\nComparison between {team1} and {team2}:")
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Metric': ['SOS', '2PA', '2P%', '3PA', '3P%', 'FTA', 'FT%', 'Turnover Margin', 'Rebound Margin', 'Pace'],
            team1: [
                team1_data['team_SOS'].values[0],
                team1_data['team_2PA'].values[0],
                team1_data['team_2P%'].values[0],
                team1_data['team_3PA'].values[0],
                team1_data['team_3P%'].values[0],
                team1_data['team_FTA'].values[0],
                team1_data['team_FT%'].values[0],
                team1_data['turnover_margin'].values[0],
                team1_data['rebound_margin'].values[0],
                team1_data['adv_team_Pace'].values[0]
            ],
            team2: [
                team2_data['team_SOS'].values[0],
                team2_data['team_2PA'].values[0],
                team2_data['team_2P%'].values[0],
                team2_data['team_3PA'].values[0],
                team2_data['team_3P%'].values[0],
                team2_data['team_FTA'].values[0],
                team2_data['team_FT%'].values[0],
                team2_data['turnover_margin'].values[0],
                team2_data['rebound_margin'].values[0],
                team2_data['adv_team_Pace'].values[0]
            ],
            'Difference': [
                team1_data['team_SOS'].values[0] - team2_data['team_SOS'].values[0],
                team1_data['team_2PA'].values[0] - team2_data['team_2PA'].values[0],
                team1_data['team_2P%'].values[0] - team2_data['team_2P%'].values[0],
                team1_data['team_3PA'].values[0] - team2_data['team_3PA'].values[0],
                team1_data['team_3P%'].values[0] - team2_data['team_3P%'].values[0],
                team1_data['team_FTA'].values[0] - team2_data['team_FTA'].values[0],
                team1_data['team_FT%'].values[0] - team2_data['team_FT%'].values[0],
                team1_data['turnover_margin'].values[0] - team2_data['turnover_margin'].values[0],
                team1_data['rebound_margin'].values[0] - team2_data['rebound_margin'].values[0],
                team1_data['adv_team_Pace'].values[0] - team2_data['adv_team_Pace'].values[0]
            ]
        })
        
        print(comparison)
    else:
        # Display top 10 teams by 2PA
        print("\nTop 10 teams by 2-point attempts:")
        print(results.head(10)[['School', 'team_SOS', 'team_2PA']])
        
        # Display top 10 teams by turnover margin
        print("\nTop 10 teams by turnover margin:")
        print(results.sort_values('turnover_margin', ascending=False).head(10)[['School', 'team_SOS', 'turnover_margin']])
        
        # Display top 10 teams by rebound margin
        print("\nTop 10 teams by rebound margin:")
        print(results.sort_values('rebound_margin', ascending=False).head(10)[['School', 'team_SOS', 'rebound_margin']])
        
        # Display top 10 teams by pace
        print("\nTop 10 teams by pace:")
        print(results.sort_values('adv_team_Pace', ascending=False).head(10)[['School', 'team_SOS', 'adv_team_Pace']])
    
    print("\nAnalysis complete! Full results saved to team_metrics_analysis.csv")
