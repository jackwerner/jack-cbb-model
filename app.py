import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_margin, calculate_spread_probability
import pickle

def main():
    st.title("College Basketball Game Predictor")
    
    # Add description of prediction variables
    with st.expander("About the Prediction Model"):
        st.markdown("""
        This model predicts game outcomes based on the following team metrics:
        
        - **Strength of Schedule (SOS)**: Difficulty of a team's schedule
        - **2-Point Attempts (2PA)**: Average number of 2-point shots attempted per game
        - **2-Point Percentage (2P%)**: Success rate on 2-point shots
        - **3-Point Attempts (3PA)**: Average number of 3-point shots attempted per game
        - **3-Point Percentage (3P%)**: Success rate on 3-point shots
        - **Free Throw Attempts (FTA)**: Average number of free throws attempted per game
        - **Free Throw Percentage (FT%)**: Success rate on free throws
        - **Turnover Margin**: Difference between turnovers forced and committed
        - **Rebound Margin**: Difference between rebounds collected and allowed
        - **Pace**: Number of possessions per 40 minutes
        
        The model calculates the difference in these metrics between the two teams to predict the final margin.
        
        """)
        
        # Add model variable importance visualization
        st.subheader("Model Feature Importance")
        
        # Load model to get feature importance
        with open('margin_predictor_xgboost_neg_root_mean_squared_error.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        
        # Get feature importance
        feature_names = [
            'SOS Difference', '2PA Difference', '2P% Difference', 
            '3PA Difference', '3P% Difference', 'FTA Difference',
            'FT% Difference', 'Turnover Margin Difference', 
            'Rebound Margin Difference', 'Pace Difference'
        ]
        
        importance = model.feature_importances_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Display bar chart
        st.bar_chart(importance_df.set_index('Feature'))
        
        st.markdown("""
        The chart above shows the relative importance of each feature in the prediction model.
        Higher values indicate features that have more influence on the predicted margin.
        """)
    
    # Load team names for dropdowns
    metrics_df = pd.read_csv('team_metrics_analysis.csv')
    team_list = sorted(metrics_df['School'].unique())
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team 1")
        team1 = st.selectbox("Select first team", team_list, key="team1")
    
    with col2:
        st.subheader("Team 2")
        team2 = st.selectbox("Select second team", team_list, key="team2")
    
    # Market spread input
    market_spread = st.number_input(
        "Enter market spread from Team 1's perspective (negative means Team 1 is favored)",
        value=0.0,
        step=0.5,
        help="Example: If Team 1 is a 5.5 point favorite, enter: -5.5\nIf Team 1 is a 5.5 point underdog, enter: +5.5"
    )
    
    if st.button("Predict Game"):
        try:
            # Calculate prediction
            margin, explanation = predict_margin(team1, team2)
            prob_over = calculate_spread_probability(float(margin), float(market_spread))
            
            # Display results in an expander
            with st.expander("Game Prediction Results", expanded=True):
                # Margin prediction
                st.markdown(f"**Predicted margin ({team1} vs {team2}):** {margin:.1f}")
                if margin > 0:
                    st.markdown(f"🏀 **{team1}** is predicted to win by **{abs(margin):.1f}** points")
                else:
                    st.markdown(f"🏀 **{team2}** is predicted to win by **{abs(margin):.1f}** points")
                
                # Spread analysis
                st.markdown("---")
                st.markdown(f"**Market spread:** {market_spread:+.1f} ({team1}'s perspective)")
                
                if market_spread < 0:  # Team1 is favored
                    st.markdown(f"**{team1}** is favored by {abs(market_spread)} points")
                    st.markdown("\n")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{team1} cover probability", 
                                f"{prob_over*100:.1f}%",
                                help=f"Probability {team1} wins by > {abs(market_spread)}")
                    with col2:
                        st.metric(f"{team2} cover probability", 
                                f"{(1-prob_over)*100:.1f}%",
                                help=f"Probability {team2} loses by < {abs(market_spread)}")
                else:
                    st.markdown(f"**{team2}** is favored by {market_spread} points")
                    st.markdown("\n")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{team1} cover probability", 
                                f"{prob_over*100:.1f}%",
                                help=f"Probability {team1} loses by < {market_spread}")
                    with col2:
                        st.metric(f"{team2} cover probability", 
                                f"{(1-prob_over)*100:.1f}%",
                                help=f"Probability {team2} wins by > {market_spread}")
                
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
