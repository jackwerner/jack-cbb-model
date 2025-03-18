import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm

def load_model_and_scaler(model_path='margin_predictor_xgboost_neg_root_mean_squared_error.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler']

def predict_margin(team1, team2):
    # Load the metrics data
    metrics_df = pd.read_csv('team_metrics_analysis.csv')
    
    # Load the model and scaler
    model, scaler = load_model_and_scaler()
    
    # Get team metrics
    team1_metrics = metrics_df[metrics_df['School'].str.lower() == team1.lower()]
    team2_metrics = metrics_df[metrics_df['School'].str.lower() == team2.lower()]
    
    # Check if both teams exist in the dataset
    if team1_metrics.empty:
        raise ValueError(f"Team '{team1}' not found in metrics data")
    if team2_metrics.empty:
        raise ValueError(f"Team '{team2}' not found in metrics data")
    
    # Create feature vector: difference between team metrics
    features = [
        team1_metrics['team_SOS'].values[0] - team2_metrics['team_SOS'].values[0],
        team1_metrics['team_2PA'].values[0] - team2_metrics['team_2PA'].values[0],
        team1_metrics['team_2P%'].values[0] - team2_metrics['team_2P%'].values[0],
        team1_metrics['team_3PA'].values[0] - team2_metrics['team_3PA'].values[0],
        team1_metrics['team_3P%'].values[0] - team2_metrics['team_3P%'].values[0],
        team1_metrics['team_FTA'].values[0] - team2_metrics['team_FTA'].values[0],
        team1_metrics['team_FT%'].values[0] - team2_metrics['team_FT%'].values[0],
        team1_metrics['turnover_margin'].values[0] - team2_metrics['turnover_margin'].values[0],
        team1_metrics['rebound_margin'].values[0] - team2_metrics['rebound_margin'].values[0],
        team1_metrics['adv_team_Pace'].values[0] - team2_metrics['adv_team_Pace'].values[0]
    ]
    
    # Scale the features
    X_scaled = scaler.transform(np.array(features).reshape(1, -1))
    
    # Make prediction
    predicted_margin = model.predict(X_scaled)[0]
    
    return predicted_margin

def calculate_spread_probability(predicted_margin, market_spread, rmse=10.88276):
    """
    Calculate probability that actual margin will be higher/lower than market spread.
    
    Args:
        predicted_margin: Model's predicted margin (positive means team1 wins by that much)
        market_spread: Market spread from team1's perspective 
                      (negative means team1 is favored, e.g., -3 means favored by 3)
        rmse: Root Mean Square Error from model training
    
    Returns:
        Probability that team1 will cover their spread
    """
    # For team1 to cover the spread:
    # If team1 is favored (negative spread): they need to win by more than the spread
    # If team1 is underdog (positive spread): they need to lose by less than the spread or win outright
    
    # Calculate how many points team1 needs to outperform our prediction by to cover the spread
    margin_difference = -market_spread - predicted_margin
    
    # Calculate z-score
    z_score = margin_difference / rmse
    
    # Calculate probability of covering (probability that actual margin > market spread)
    prob_cover = 1 - norm.cdf(z_score)
    
    # Add debug prints
    print(f"Predicted margin: {predicted_margin}")
    print(f"Market spread: {market_spread}")
    print(f"Margin difference: {margin_difference}")
    print(f"Z-score: {z_score}")
    
    return prob_cover

if __name__ == "__main__":
    # Example usage

    # Example: If Team 1 is a 5.5 point favorite, enter: -5.5
    # Example: If Team 1 is a 5.5 point underdog, enter: +5.5
    
    team1 = "Duke"
    team2 = "Florida"
    market_spread = 2.5
    
    try:
        margin = predict_margin(team1, team2)
        prob_cover = calculate_spread_probability(float(margin), float(market_spread))
        
        # Display game prediction
        print(f"\nPredicted margin ({team1} vs {team2}): {margin:.1f}")
        if margin > 0:
            print(f"{team1} is predicted to win by {abs(margin):.1f} points")
        else:
            print(f"{team2} is predicted to win by {abs(margin):.1f} points")
        
        # Display spread analysis
        print(f"\nMarket spread: {market_spread:+.1f} ({team1}'s perspective)")
        if market_spread < 0:  # Negative spread means team1 is favored
            print(f"{team1} is favored by {abs(market_spread)} points")
            print("\n")
            print(f"Probability {team1} covers (wins by > {abs(market_spread)}): {prob_cover*100:.1f}%")
            print(f"Probability {team2} covers (loses by < {abs(market_spread)}): {(1-prob_cover)*100:.1f}%")
        else:
            print(f"{team2} is favored by {market_spread} points")
            print("\n")
            print(f"Probability {team1} covers (loses by < {market_spread}): {prob_cover*100:.1f}%")
            print(f"Probability {team2} covers (wins by > {market_spread}): {(1-prob_cover)*100:.1f}%")
        
    except ValueError as e:
        print(f"Error: {e}")
