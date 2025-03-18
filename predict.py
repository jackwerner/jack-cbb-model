import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm

def load_model_and_scaler(model_path='margin_predictor_xgboost_neg_root_mean_squared_error.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler']

def predict_margin(team1, team2, explain=False):
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
    
    # Feature names for explanation
    feature_names = [
        'SOS_diff', '2PA_diff', '2P%_diff', '3PA_diff', '3P%_diff',
        'FTA_diff', 'FT%_diff', 'turnover_margin_diff', 'rebound_margin_diff', 'pace_diff'
    ]
    
    # Create feature vector: difference between team metrics (team1 - team2)
    features_1_2 = [
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
    
    # Create feature vector in reverse order: (team2 - team1)
    features_2_1 = [-x for x in features_1_2]
    
    # Create a dictionary of features for explanation
    feature_dict = dict(zip(feature_names, features_1_2))
    
    # Scale the features for both directions
    X_scaled_1_2 = scaler.transform(np.array(features_1_2).reshape(1, -1))
    X_scaled_2_1 = scaler.transform(np.array(features_2_1).reshape(1, -1))
    
    # Make predictions in both directions
    predicted_margin_1_2 = model.predict(X_scaled_1_2)[0]
    predicted_margin_2_1 = model.predict(X_scaled_2_1)[0]
    
    # Average the predictions (negate the second prediction since it's from team2's perspective)
    predicted_margin = (predicted_margin_1_2 - predicted_margin_2_1) / 2
    
    # Generate explanation if requested
    explanation = None
    if explain:
        # Get feature importances from the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            top_features = [(feature_names[i], features_1_2[i], importances[i]) 
                           for i in sorted_idx[:3]]  # Top 3 most important features
            
            explanation = {
                'top_features': top_features,
                'all_features': feature_dict,
                'team1_metrics': team1_metrics.iloc[0].to_dict(),
                'team2_metrics': team2_metrics.iloc[0].to_dict(),
                'raw_predictions': {
                    'team1_vs_team2': predicted_margin_1_2,
                    'team2_vs_team1': -predicted_margin_2_1
                }
            }
    
    return predicted_margin, explanation

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

def explain_prediction(explanation, team1, team2):
    """Generate a human-readable explanation for the prediction"""
    if not explanation:
        return "No explanation available."
    
    top_features = explanation['top_features']
    
    explanation_text = f"Why {team1} vs {team2} prediction:\n\n"
    
    for feature_name, value, importance in top_features:
        # Format the feature name for display
        display_name = feature_name.replace('_diff', '').replace('_', ' ').title()
        
        # Determine which team has the advantage for this feature
        if value > 0:
            advantage = f"{team1} has a {display_name} advantage"
        elif value < 0:
            advantage = f"{team2} has a {display_name} advantage"
        else:
            advantage = f"Both teams are equal in {display_name}"
        
        # Add to explanation
        explanation_text += f"- {advantage} ({abs(value):.2f})\n"
    
    # Add overall summary
    explanation_text += "\nOther factors considered: shooting percentages, pace, turnovers, and rebounding."
    
    return explanation_text

def print_team_features(explanation, team1, team2):
    """Print all features for both teams"""
    if not explanation:
        return "No feature data available."
    
    team1_metrics = explanation['team1_metrics']
    team2_metrics = explanation['team2_metrics']
    
    print("\n" + "="*50)
    print("TEAM METRICS COMPARISON")
    print("="*50)
    
    # Get all metrics keys, excluding 'School' and any other non-metric fields
    exclude_keys = ['School', 'Conf', 'G']
    metric_keys = [k for k in team1_metrics.keys() if k not in exclude_keys]
    
    # Print header
    print(f"{'Metric':<25} {team1:<15} {team2:<15} {'Difference':<10}")
    print("-" * 65)
    
    # Print each metric
    for key in metric_keys:
        if isinstance(team1_metrics[key], (int, float)) and isinstance(team2_metrics[key], (int, float)):
            val1 = team1_metrics[key]
            val2 = team2_metrics[key]
            diff = val1 - val2
            
            # Format the display name
            display_name = key.replace('team_', '').replace('adv_team_', '')
            
            # Format the values based on their magnitude
            if abs(val1) < 0.1 or abs(val2) < 0.1:
                print(f"{display_name:<25} {val1:<15.4f} {val2:<15.4f} {diff:<+10.4f}")
            else:
                print(f"{display_name:<25} {val1:<15.2f} {val2:<15.2f} {diff:<+10.2f}")

if __name__ == "__main__":
    # Example usage

    # Example: If Team 1 is a 5.5 point favorite, enter: -5.5
    # Example: If Team 1 is a 5.5 point underdog, enter: +5.5
    
    team1 = "Alabama"
    team2 = "Robert Morris"
    market_spread = 0
    
    try:
        margin, explanation = predict_margin(team1, team2, explain=True)
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
        
        # Display explanation
        print("\n" + "="*50)
        print("PREDICTION EXPLANATION")
        print("="*50)
        print(explain_prediction(explanation, team1, team2))
        
        # Print all features for both teams
        print_team_features(explanation, team1, team2)
        
    except ValueError as e:
        print(f"Error: {e}")
