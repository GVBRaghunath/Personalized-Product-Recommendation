from flask import Flask, request,render_template ,render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta 


app = Flask(__name__,static_folder='static')

# Load the user data and product data
user_data = pd.read_csv("user_data.csv")
product_data = pd.read_csv("product_data.csv")

# Merge the user and product data into a single dataset.
data = user_data.merge(product_data, on="product_id")

# Convert interaction_type to a numerical value (purchase: 2, wishlist: 1, view: 0)
data["interaction_type"] = data["interaction_type"].apply(lambda x: 2 if x == "purchase" else (1 if x == "wishlist" else 0))

# Create a dictionary to map product IDs to indices for embedding lookup.
product_id_to_index = {product_id: index for index, product_id in enumerate(data["product_id"].unique())}

# Filter products by popularity and date
current_year = datetime.now().year
five_years_ago = datetime.now() - timedelta(days=5*365)  # Calculate the date five years ago
five_years_ago_year = five_years_ago.year

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form.get("user_id")
        num_recommendation=request.form.get('num_recommendations')
        while True:
            name = user_name
            name = name.upper()
            output=user_name.upper()
              # Convert to uppercase to handle both uppercase and lowercase inputs
            recommended_product_ids = []
            if "CUST0001" <= name <= "CUST0150" and name in user_data["user_id"].values:
                num_recommendations = int(num_recommendation)
                recommended_product_ids = recommend_products(name, num_recommendations)
            
                high_priority_recs = []
                wishlst_priority_recs = []
                low_priority_recs = []
                hghidx, lowidx ,wshidx = 1, 1, 1
                                
                # purchase_explanation = "based on the product purchased, with price equal to or greater than,interaction type with value 2"
                # view_explanation = "based on the product viewed, with price equal to or less than,interaction type with value 0"
                # wishlist_explanation = "based on the interaction type with value 1"
                
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
            recommendations=all_priority_recs
            return render_template('index.html', recommendations=recommendations)
        
        
    return '''
        <head>
        
            <style>
                *{
                    margin:0px;
                    padding:0px;
                    font-family: 'Montserrat', sans-serif;
                }
                input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        input[type="number"] {
            -moz-appearance: textfield;
        }
                .first{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction:column;
                    height:100vh;
                    width:100vw;
                    background-color: #006cb4;
                }
                .inner{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height:20vh;
                    width:100vw;
                }
                .form{
                    display: flex;
                    justify-content: space-between;
                    flex-direction: column;
                    align-items:space-evenly;
                    align-items: center;
                    height:80vh;
                    width:100vw;
                }
                input{
                    height:4vh;
                    background-color: transparent;
                    outline: none;
                    border:2px solid black;
                    border-radius: 4px;
                    font-weight: 600;
                    color:black;
                    padding-left: 2%;
                }
                input[type="number"]{
                    width:14vw
                }
                input:focus{
                    border:2px solid #5866cf;
                }
                input[type="submit"]
                {
                    height:5vh;
                    width:13vw;
                    background-color: #027cd5;
                    color:white;
                    border:none;
                }
                input[type="submit"]:hover{
                    background-color:#cee5f6;
                    color:black;
                }
                ::placeholder{
                    padding-left:18% ;
                    color:black
                    
                }
                form{
                    display: flex;
                    justify-content: space-evenly;
                    align-items: center;
                    flex-direction: column;
                    height:60%;
                    width:30%;
                    border-radius: 10px;
                    background-color: rgb(254, 254, 22);
                    border-top:1px solid rgba(255, 255, 255, 0.637);
                    border-left:1px solid rgba(255, 255, 255, 0.603);
                    /* border:2px solid red; */
                }
                .f{background-color:yellow;
                height:vh;
                width:100%;
                text-align:center;
                }
                .cr{
                    height:60vh;
                    width:60vh;
                    background-image:logo.png;
                }
                .forgot{
                    font-size: 12px;
                    color:black;
                    font-weight: 600;
                }
                #num_recommendations{
                    width:22vw;
                }
                @media (max-width:800px)
                {
                    form{
                        width:80%;
                    }
                    input[type="submit"]
                    {
                    height:5vh;
                    width:18vw;
                    }
                    .cr{
                        height:40vh;
                        width:60vw;
                    }
                }
            </style>
        </head>
        <body>
            
            <div class="first">

                <div class="form">
                    <h1 class="f" >Personalized Product Recommendations</h1>
                    <form method="POST">
                        Recommendation System
                        <input type="text" type="text" id="user_id" name="user_id" required placeholder="Enter Customer Id">
                        <input type="number" id="num_recommendations" name="num_recommendations" required min="0" max="50" step="5" placeholder="Enter required Recommendations">
                        <input type="submit" value="Get Recommendations">
                    </form>
                </div>
            </div>
        </body>
    '''

if __name__ == '__main__':
    app.run(debug=True)