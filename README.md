# Stock-Movement-Analysis-Based-on-Social-Media-Sentiment

## Objective

This project aims to develop a machine learning model that predicts stock movements by analyzing sentiment in social media discussions (Reddit posts). The model is trained using sentiment analysis and other features extracted from Reddit posts to predict stock price trends.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.x
- `yfinance` - For fetching stock data
- `pandas` - For data manipulation
- `numpy` - For numerical operations
- `praw` - For scraping Reddit posts
- `scikit-learn` - For machine learning models
- `textblob` - For sentiment analysis
- `joblib` - For saving the trained model

## Setup

1. Install the required libraries using pip:
   ```bash
   pip install yfinance pandas numpy praw scikit-learn textblob joblib
   
2. Reddit API Credentials: You need to create a Reddit account and register for API access at Reddit API. After registration, replace YOUR_CLIENT_ID and YOUR_CLIENT_SECRET in the code with your actual Reddit API credentials.

3. Download the code: Save the provided Python code to your working directory.

## Running the Code
1. Open a terminal or command prompt in your project directory.

2. Run the Python script to start the process:
      ```
     python model.py ```
3.The script will:

  - Scrape recent Reddit posts from the wallstreetbets subreddit.
  - Fetch stock data for a specified ticker (e.g., TSLA).
  - Preprocess the data, including sentiment analysis and feature extraction.
  - Train a machine learning model (Random Forest) to predict stock movements.
  - Print the model's performance metrics, including:
      - Accuracy
      - Precision
      - Recall
      - F1 Score
  - Provide a detailed classification report.
    
## Model Evaluation
The script will print out the following evaluation metrics:

- Accuracy: The proportion of correct predictions.
- Precision: The proportion of true positive predictions out of all positive predictions.
- Recall: The proportion of true positive predictions out of all actual positive cases.
- F1 Score: The harmonic mean of precision and recall.
- Classification Report: Detailed metrics for each class (0 for no movement, 1 for price increase).
  
## Saving the Model
After training, the model will be saved as stock_movement_model.pkl using joblib. This model can be reloaded for future predictions.

## Future Improvements
- Use more advanced sentiment analysis models: Implement more sophisticated models (e.g., BERT or GPT) for more accurate sentiment analysis.
- Incorporate other social media platforms: Extend the project by scraping and analyzing data from platforms like Twitter or Telegram.
- Fine-tune model hyperparameters: Experiment with hyperparameter tuning for the Random Forest model to improve accuracy.

