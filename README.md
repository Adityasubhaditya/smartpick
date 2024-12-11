# Smartpick(Mobile Sales Data Analysis and Recommendation System)

## Overview
This project leverages machine learning and data analysis techniques to provide phone recommendations based on user preferences and train a classification model to categorize phone prices. The dataset used contains information about mobile phones available on Flipkart, including details such as price, RAM, storage, camera specifications, and customer ratings.

## Project Features
- **Data Preprocessing**: Handles missing values, converts data types, and standardizes data for analysis.
- **Phone Recommendation Engine**: Uses Euclidean distance to find the top 5 phones matching user preferences.
- **Random Forest Model**: Classifies phones into different price ranges using a supervised machine learning approach.
- **Hyperparameter Tuning**: Employs GridSearchCV to optimize model performance.

## Technologies Used
- **Programming Language**: Python 3.x
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `scikit-learn` for machine learning and preprocessing
- **Tools**:
  - Jupyter Notebook/IDE for development

## Data Input
Actual price
Discount price
RAM (GB)
Storage (GB)
Camera
Stars

## User Interaction
Input your preferences when prompted to receive phone recommendations based on price, RAM, storage, display size, and camera megapixels.
Model Training and Evaluation
The script will train a Random Forest Classifier to predict the price range of phones and print an accuracy score and a classification report.

## Example Output
Top 5 Recommended Phones:
A table displaying the recommended phones that match user preferences.
Classification Report:
Model performance metrics, including precision, recall, and F1-score for each price range.
Best Hyperparameters:
The optimal parameters used for the Random Forest model.

## Future Enhancements
Integration with a Web Application: Build a web-based UI using frameworks like Flask or Streamlit for interactive user input.
Advanced Recommendation Algorithms: Implement collaborative filtering for better user experience.
Real-time Data Integration: Fetch data dynamically from APIs for up-to-date analysis.

For questions or contributions, reach out to:

Email: adityasubhaditya@gmail.com
