import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from texttable import Texttable

# Load the user data and product data
user_data = pd.read_csv("user_data.csv")
product_data = pd.read_csv("product_data.csv")

# Merge the user and product data into a single dataset.
data = user_data.merge(product_data, on="product_id")

# Convert interaction_type to a numerical value (purchase: 2, wishlist: 1, view: 0)
data["interaction_type"] = data["interaction_type"].apply(lambda x: 2 if x == "purchase" else (1 if x == "wishlist" else 0))

# Create a dictionary to map product IDs to indices for embedding lookup.
product_id_to_index = {product_id: index for index, product_id in enumerate(data["product_id"].unique())}

# Train the reinforcement learning model.
num_classes = len(product_id_to_index)  # Number of unique products

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=num_classes, output_dim=16, input_length=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),  # Use softmax activation for multi-class output
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Prepare input data for training
product_indices = data["product_id"].map(product_id_to_index)
user_ratings = pd.get_dummies(data["product_id"].map(product_id_to_index))  # One-hot encoding for multi-class

model.fit(product_indices, user_ratings, epochs=10)  # Use user_ratings as target data

# Make recommendations for users.
def recommend_products(user_id, num_recommendations):
    # Get the user's interactions (purchases, wishlists and views) with products.
    user_interactions = data[data["user_id"] == user_id]
    user_product_indices = user_interactions["product_id"].map(product_id_to_index).values
    
    # Predict user's preference for products.
    predicted_preferences = model.predict(user_product_indices)
    
    # Combine predicted preferences with product indices, popularity, date, and price.
    product_scores = np.column_stack((user_product_indices, predicted_preferences, 
                                      user_interactions["popularity"].astype(float), 
                                      pd.to_datetime(user_interactions["date"], format="%d-%m-%Y").dt.year,
                                      user_interactions["price"].astype(float)))
    
    # Sort products by predicted preference, popularity, date, and price.
    sorted_products = product_scores[np.lexsort((product_scores[:, 1], -product_scores[:, 2], -product_scores[:, 3], product_scores[:, 4]))[::-1]]
    
    # Select top N unique product indices
    unique_indices = []
    recommended_products = []
    for idx in sorted_products[:, 0]:
        if idx not in unique_indices:
            unique_indices.append(idx)
            recommended_products.append(idx)
            if len(recommended_products) >= num_recommendations:
                break
    
    # Map unique indices to product IDs
    recommended_product_ids = [product_id for index in recommended_products 
                               if (product_id := next((pid for pid, idx in product_id_to_index.items() if idx == index), None))]
    
    return recommended_product_ids

# Usage loops
while True:
    print("Available Customer IDs: CUST0001 to CUST0150")
    name = input("Enter your customer ID: ")
    
    if name.lower() == 'quit':
        print("------------------------------------ Exiting recommendation system. ------------------------------------\n")
        break
    
    name = name.upper()  # Convert to uppercase to handle both uppercase and lowercase inputs
    recommended_product_ids = []
    if "CUST0001" <= name <= "CUST0150" and name in user_data["user_id"].values:
        print("Maximum Limit for Recommendations is: 70")
        num_recommendations = int(input("Enter the number of recommendations needed: "))
        if 1 <= num_recommendations <= 70:
            recommended_product_ids = recommend_products(name, num_recommendations)
    
            high_priority_recs = []
            wishlst_priority_recs = []
            low_priority_recs = []
            hghidx, lowidx, wshidx = 1, 1, 1
            
            # Get purchased product price
            purchased_products = user_data[(user_data["user_id"] == name) & (user_data["interaction_type"] == 2)]
            if not purchased_products.empty:
                purchased_product_price = purchased_products["price"].iloc[0]
            else:
                purchased_product_price = 0.0  # Default value if no purchase
            
            # Get viewed product prices
            viewed_products = user_data[(user_data["user_id"] == name) & (user_data["interaction_type"] == 0)]
            if not viewed_products.empty:
                viewed_product_min_price = viewed_products["price"].min()
                viewed_product_max_price = viewed_products["price"].max()
            else:
                viewed_product_min_price = 0.0  # Default value if no viewed products
                viewed_product_max_price = 0.0  # Default value if no viewed products
            
            for idx, product_id in enumerate(recommended_product_ids, start=1):
                product_info = data[data["product_id"] == product_id]
                interaction_type = product_info["interaction_type"].values[0]
                price = float(product_info["price"].iloc[0])
                popularity = product_info["popularity"].iloc[0]
                date = product_info["date"].iloc[0]
                
                if interaction_type == 2:
                    high_priority_recs.append([hghidx, product_id[:60], f"Price: {price:.2f} -> Popularity: {popularity} -> Date: {date}"])
                    hghidx += 1
                elif interaction_type == 0 and price >= viewed_product_min_price and price <= viewed_product_max_price:
                    low_priority_recs.append([lowidx, product_id[:60], f"Price: {price:.2f} -> Popularity: {popularity} -> Date: {date}"])
                    lowidx += 1
                elif interaction_type == 0 and price > purchased_product_price:
                    low_priority_recs.append([lowidx, product_id[:60], f"Price: {price:.2f} -> Popularity: {popularity} -> Date: {date}"])
                    lowidx += 1
            
            wshidx = lowidx
            for idx, product_id in enumerate(recommended_product_ids, start=1):
                product_info = data[data["product_id"] == product_id]
                interaction_type = product_info["interaction_type"].values[0]
                price = float(product_info["price"].iloc[0])
                popularity = product_info["popularity"].iloc[0]
                date = product_info["date"].iloc[0]
                    
                if interaction_type == 1:
                    wishlst_priority_recs.append([wshidx, product_id[:60], f"Price: {price:.2f} -> Popularity: {popularity} -> Date: {date}"])
                    wshidx += 1
            
            all_priority_recs = high_priority_recs + wishlst_priority_recs + low_priority_recs
            all_priority_recs.sort(key=lambda x: x[0])
            rank=1
            for lst in all_priority_recs:
                lst[0]=rank
                rank+=1
            t = Texttable()
            header = ["Rank", "     Product ID (Product names are truncated to 60 characters)    ", "       Filtered Through          "]
            t.add_row(header)
            for rec in all_priority_recs:
                t.add_row(rec)
            print("\nFor customer:", name, "Recommended products are:")
            print("\n--------------------- Product preferences are made based on predicted prefernce interaction type, price, popularity, and date. ---------------------\n")
            print(t.draw())
            print("\nNote: Number of recommendations may be lower than specified because only optimal recommendations of the products are recommended\n\n-------------- Type Quit to EXIT --------------\n")
        else:
            print("*********************** Recommendation Limit is up to 70 only ***********************")

    else:
        print("********* Invalid customer ID. Please enter a valid ID within the range CUST0001 to CUST0150. *********\n")