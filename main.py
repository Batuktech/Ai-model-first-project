import pandas as pd
import numpy as np
import os
import argparse
import logging
from datetime import datetime

# Import modules
from data_collection import collect_event_data, save_data, load_data
from data_analysis import calculate_metrics, analyze_trends
from ml_prediction import train_view_prediction_model, predict_views, visualize_predictions
from simple_viewer import view_results

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YouTube Famous Events Analyzer')
    
    parser.add_argument('--collect-data', action='store_true',
                       help='Force data collection even if local data exists')
    
    parser.add_argument('--api-key', type=str, default=None,
                       help='YouTube API key (overrides config)')
    
    parser.add_argument('--events', type=str, nargs='+',
                       help='List of events to analyze (overrides config)')
    
    parser.add_argument('--results-per-event', type=int, default=None,
                       help='Number of results per event (overrides config)')
    
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning for faster model training')
    
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    return parser.parse_args()

def main():
    """Main function for the YouTube Famous Events Analyzer"""
    print("\nYouTube Famous Events View Counter & Analyzer")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Import config settings
    try:
        import config
        api_key = args.api_key or config.YOUTUBE_API_KEY
        events = args.events or config.FAMOUS_EVENTS
        results_per_event = args.results_per_event or config.RESULTS_PER_EVENT
        data_file = config.DATA_FILE
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading config: {e}")
        # Set defaults if config not available
        api_key = args.api_key
        events = args.events or ["World Cup", "Super Bowl", "NBA Finals", "Olympics", "Grammy Awards"]
        results_per_event = args.results_per_event or 30
        data_file = "youtube_events_data.csv"
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we need to collect data
    data = None
    if not args.collect_data and os.path.exists(data_file):
        print(f"Loading existing data from {data_file}...")
        data = load_data(data_file)
        print(f"Loaded {len(data)} videos")
    
    # Collect data if needed
    if data is None or data.empty or args.collect_data:
        if api_key is None or api_key == "YOUR_API_KEY_HERE":
            logger.error("YouTube API key not provided. Use --api-key or set in config.py")
            return
        
        print(f"Collecting data for events: {', '.join(events)}...")
        print(f"Getting up to {results_per_event} videos per event")
        
        data = collect_event_data(api_key, events, results_per_event)
        
        if data is None or data.empty:
            logger.error("Data collection failed. Check API key and connection.")
            return
        
        save_data(data, data_file)
        print(f"Collected and saved {len(data)} videos to {data_file}")
    
    # Basic data validation
    if 'event' not in data.columns or 'view_count' not in data.columns:
        logger.error("Data missing required columns. Check data format.")
        return
    
    if len(data) < 10:
        logger.warning("Very small dataset. Results may be unreliable.")
    
    # Print basic dataset stats
    print("\nDataset Overview:")
    print(f"Total videos: {len(data)}")
    print(f"Events covered: {', '.join(data['event'].unique())}")
    print(f"Date range: {data['published_at'].min()} to {data['published_at'].max()}")
    print(f"Total views analyzed: {data['view_count'].sum():,}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(data)
    
    # Print top 3 events by total views
    top_events = metrics['event_total_views'].head(3)
    print("\nTop events by total views:")
    for _, row in top_events.iterrows():
        print(f"  {row['event']}: {row['view_count']:,} views")
    
    # Analyze trends
    print("\nAnalyzing trends and patterns...")
    trends = analyze_trends(data)
    
    # Print key correlations
    print(f"Correlation between video age and views: {trends['age_view_correlation']:.2f}")
    print(f"Correlation between likes and views: {trends['like_view_correlation']:.2f}")
    print(f"Correlation between comments and views: {trends['comment_view_correlation']:.2f}")
    
    # Train the model
    print("\nTraining view prediction model...")
    model = train_view_prediction_model(data, tune_hyperparameters=not args.no_tune)
    
    # Make predictions for top videos
    print("\nGenerating predictions for top videos:")
    
    # Get top video for each event
    for event in data['event'].unique():
        event_data = data[data['event'] == event]
        top_video = event_data.sort_values('view_count', ascending=False).iloc[0]
        
        print(f"\n{event} - Top Video: {top_video['title'][:50]}...")
        print(f"  Current views: {top_video['view_count']:,}")
        print(f"  Days since publishing: {top_video['days_since_publishing']}")
        
        # Predict views
        results = predict_views(
            model,
            event,
            top_video['days_since_publishing'],
            top_video['view_count'],
            top_video['like_count'],
            top_video['comment_count'],
            data,
            days_back=30,
            days_forward=30
        )
        
        # Calculate future growth
        future_data = results[results['is_future']]
        if not future_data.empty:
            final_views = future_data['predicted_views'].iloc[-1]
            growth = final_views - top_video['view_count']
            growth_percent = (growth / top_video['view_count']) * 100
            
            print(f"  Predicted views in 30 days: {final_views:,.0f}")
            print(f"  Expected growth: +{growth:,.0f} views ({growth_percent:.1f}%)")
        
        # Visualize and save predictions
        plot_file = os.path.join(output_dir, f"prediction_{event.replace(' ', '_')}.png")
        visualize_predictions(
            results,
            title=f"View Prediction for {event} - Top Video",
            save_path=plot_file
        )
        print(f"  Prediction chart saved to {plot_file}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive HTML report...")
    try:
        view_results(data, metrics, model)
        print("Report generated and opened in browser.")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        print("Error generating report. See logs for details.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()