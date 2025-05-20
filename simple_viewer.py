import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import logging
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def train_view_prediction_model(data, tune_hyperparameters=True, cv=5):
    """
    Train the RandomForest model for predicting video views
    
    Args:
        data: DataFrame containing YouTube video data
        tune_hyperparameters: Whether to perform hyperparameter tuning
        cv: Number of cross-validation folds
    
    Returns:
        Trained model object
    """
    logger.info("Starting model training process")
    start_time = datetime.now()
    
    # Prepare features
    categorical_features = ['event']
    numeric_features = ['days_since_publishing', 'like_count', 'comment_count']
    
    # Check for missing values
    missing_values = data[categorical_features + numeric_features].isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected: {missing_values[missing_values > 0]}")
    
    # Enhanced preprocessing with imputation for robustness
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # RobustScaler is better for data with outliers
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare X and y
    X = data[categorical_features + numeric_features]
    y = data['view_count']
    
    # Check for outliers in the target variable
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outlier_mask = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
    if outlier_mask.sum() > 0:
        logger.info(f"Detected {outlier_mask.sum()} outliers in target variable")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Create base pipeline
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter tuning if requested
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning")
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        
        model = GridSearchCV(
            base_pipeline, 
            param_grid, 
            cv=cv, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        logger.info(f"Best parameters: {model.best_params_}")
    else:
        logger.info("Training with default hyperparameters")
        model = base_pipeline
        model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate multiple metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation results
    logger.info(f"Model Evaluation:")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R²: {r2:.4f}")
    
    # Feature importance analysis
    if hasattr(model, 'best_estimator_'):
        rf_model = model.best_estimator_.named_steps['regressor']
    else:
        rf_model = model.named_steps['regressor']
    
    # Log top feature importances
    logger.info("Top important features:")
    for i in range(min(10, len(rf_model.feature_importances_))):
        logger.info(f"Feature {i+1}: Importance {rf_model.feature_importances_[i]:.4f}")
    
    # Save the model
    joblib.dump(model, 'youtube_view_prediction_model.joblib')
    logger.info("Model saved to youtube_view_prediction_model.joblib")
    
    training_time = datetime.now() - start_time
    logger.info(f"Model training completed in {training_time}")
    
    return model

def predict_views(model, event, current_days, current_views, current_likes, 
                 current_comments, historical_data, days_back=30, days_forward=30):
    """
    Predict video views for a specific event
    
    Args:
        model: Trained prediction model
        event: Event name
        current_days: Current days since publishing
        current_views: Current view count
        current_likes: Current like count
        current_comments: Current comment count
        historical_data: DataFrame with historical data for comparison
        days_back: Number of days to look back
        days_forward: Number of days to predict forward
    
    Returns:
        DataFrame with prediction results
    """
    # Define the range of days for prediction
    start_day = max(1, current_days - days_back)
    end_day = current_days + days_forward
    all_days = list(range(start_day, end_day + 1))
    
    # Create feature DataFrame for prediction
    prediction_data = pd.DataFrame({
        'event': [event] * len(all_days),
        'days_since_publishing': all_days,
        'like_count': [current_likes] * len(all_days),
        'comment_count': [current_comments] * len(all_days)
    })
    
    # Predict views using the trained model
    if hasattr(model, 'predict'):
        predicted_views = model.predict(prediction_data)
    else:
        # Handle case for GridSearchCV
        predicted_views = model.best_estimator_.predict(prediction_data)
    
    # Build the result DataFrame
    results = pd.DataFrame({
        'day': all_days,
        'predicted_views': predicted_views,
        'actual_views': np.nan,
        'is_current': [d == current_days for d in all_days],
        'is_historical': [d <= current_days for d in all_days],
        'is_future': [d > current_days for d in all_days]
    })
    
    # Add actual historical view counts if available
    try:
        # Find the most similar video in historical data
        video_matches = historical_data[
            (historical_data['event'] == event) & 
            (abs(historical_data['view_count'] - current_views) < 100) & 
            (abs(historical_data['days_since_publishing'] - current_days) < 5)
        ]
        
        if len(video_matches) > 0:
            video_id = video_matches['video_id'].values[0]
            logger.info(f"Found matching video: {video_id}")
        else:
            logger.info(f"No exact match found for {event} event, using most viewed video instead")
            video_id = historical_data[historical_data['event'] == event].sort_values('view_count', ascending=False).iloc[0]['video_id']
            
    except Exception as e:
        logger.warning(f"Error finding similar video: {str(e)}")
        logger.info(f"Using most viewed video for {event} event instead")
        video_id = historical_data[historical_data['event'] == event].sort_values('view_count', ascending=False).iloc[0]['video_id']
    
    video_data = historical_data[historical_data['video_id'] == video_id]
    
    for _, row in video_data.iterrows():
        day = row['days_since_publishing']
        if day in results['day'].values:
            results.loc[results['day'] == day, 'actual_views'] = row['view_count']
    
    return results

def visualize_predictions(results, title="View Count Prediction", save_path=None):
    """
    Visualize the prediction results
    
    Args:
        results: DataFrame with prediction results
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        matplotlib.pyplot object
    """
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.plot(results['day'], results['predicted_views'], 
             label='Predicted Views', color='blue', linewidth=2)
    
    # Plot actual views where available
    actual_data = results[~results['actual_views'].isna()]
    if len(actual_data) > 0:
        plt.scatter(actual_data['day'], actual_data['actual_views'], 
                   label='Actual Views', color='green', s=50)
    
    # Mark current day
    current_day = results[results['is_current']]['day'].values[0]
    plt.axvline(x=current_day, color='red', linestyle='--', label=f'Current Day ({current_day})')
    
    # Add labels and title
    plt.xlabel('Days Since Publishing')
    plt.ylabel('View Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    return plt