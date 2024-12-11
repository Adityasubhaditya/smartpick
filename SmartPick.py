import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Step 1: Preprocess data
def preprocess_data(df):
    # Handle missing values
    df.replace('NIL', np.nan, inplace=True)
    
    # Clean price columns
    df['Actual price'] = (
        df['Actual price']
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace('', np.nan)
        .astype(float)
    )
    df['Discount price'] = (
        df['Discount price']
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace('', np.nan)
        .astype(float)
    )
    
    # Convert RAM, Storage, and Display Size to numeric
    df['RAM (GB)'] = pd.to_numeric(df['RAM (GB)'], errors='coerce')
    df['Storage (GB)'] = pd.to_numeric(df['Storage (GB)'], errors='coerce')
    df['Display Size (inch)'] = pd.to_numeric(df['Display Size (inch)'], errors='coerce')
    
    # Extract main camera megapixels
    df['Main Camera MP'] = df['Camera'].str.extract('(\d+)').astype(float)
    
    return df

# Step 2: Get phone recommendations based on user preferences
def get_phone_recommendations(df, user_preferences):
    # Features for comparison
    features = ['Actual price', 'Discount price', 'RAM (GB)', 'Storage (GB)', 
               'Display Size (inch)', 'Main Camera MP']
    
    # Create feature matrix and handle missing data
    X = df[features].fillna(df[features].mean())
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale user preferences
    user_pref_scaled = scaler.transform([user_preferences])
    
    # Calculate Euclidean distance
    distances = np.sqrt(((X_scaled - user_pref_scaled) ** 2).sum(axis=1))
    
    # Get top 5 recommendations
    top_5_indices = distances.argsort()[:5]
    recommendations = df.iloc[top_5_indices]
    
    return recommendations[['Product Name', 'Discount price', 'RAM (GB)', 'Storage (GB)', 
                             'Display Size (inch)', 'Main Camera MP', 'Stars']]

# Step 3: Machine learning model training and evaluation (Improved)
def train_random_forest(df):
    # Features for model training
    features = ['Actual price', 'Discount price', 'Stars', 'RAM (GB)', 'Storage (GB)', 'Main Camera MP']
    
    # Create a simplified target variable: Price range classification
    price_bins = [0, 10000, 20000, 30000, 50000, np.inf]
    price_labels = ['Low', 'Mid-Low', 'Mid', 'High', 'Premium']
    df['Price Range'] = pd.cut(df['Discount price'], bins=price_bins, labels=price_labels)

    # Drop rows with missing values in features or target
    df = df.dropna(subset=features + ['Price Range'])

    # Create feature matrix and target variable
    X = df[features]
    y = df['Price Range']

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Best Parameters:", grid_search.best_params_)

    return accuracy

# Main execution
if __name__ == "__main__":
    # Step 1: Read the dataset
    df = pd.read_csv("C:/Users/VICTUS/OneDrive/Desktop/Datasets/Mobiles_Sales_Flipkart.csv")
    # Step 2: Preprocess the data
    df_processed = preprocess_data(df)
    
    # Step 3: Example user preferences input
    print("Enter your preferences:")
    price_range = float(input("Maximum price (in Rs.): "))
    min_ram = float(input("Minimum RAM (in GB): "))
    min_storage = float(input("Minimum Storage (in GB): "))
    min_display = float(input("Minimum Display Size (in inches): "))
    min_camera = float(input("Minimum Camera MP: "))
    
    user_preferences = [
        price_range,  # Actual price
        price_range,  # Discount price
        min_ram,      # RAM
        min_storage,  # Storage
        min_display,  # Display Size
        min_camera    # Main Camera MP
    ]
    
    # Step 4: Get recommendations based on user preferences
    recommendations = get_phone_recommendations(df_processed, user_preferences)
    
    print("\nTop 5 Recommended Phones based on your preferences:")
    print("================================================")
    print(recommendations.to_string(index=False))
    
    # Step 5: Train and evaluate the RandomForest model and show accuracy after recommendations
    accuracy = train_random_forest(df_processed)
    print(f"\n Random Forest Model Accuracy: {accuracy:.2f}")
