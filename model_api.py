from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load everything
model = joblib.load("final_model.pkl")
encoder = joblib.load("encoder.pkl")
meal_df = pd.read_csv("meal_dishes_final_corrected.csv")

# Feature separation (based on your training)
categorical_columns = [
    'Gender', 'Weight_Goal', 'Health_Condition',
    'User_Diet_Type', 'Activity_Level', 'Meal_Type', 'Dish_Diet_Type'
]

numerical_columns = [
    'Age', 'Calories', 'Protein', 'Carbs', 'Fat', 'Fiber'
]

all_columns = numerical_columns + categorical_columns

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_input_dict = request.get_json()
        meal_types = ["Breakfast", "Lunch", "Snack", "Dinner"]
        final_recommendations = {}

        for meal in meal_types:
            # Filter dishes by current meal type
            filtered_df = meal_df[meal_df["Meal_Type"] == meal]
            rows = []

            for idx, dish in filtered_df.iterrows():
                row = {
                    'Age': user_input_dict['Age'],
                    'Gender': user_input_dict['Gender'],
                    'Weight_Goal': user_input_dict['Weight_Goal'],
                    'Health_Condition': user_input_dict['Health_Condition'],
                    'User_Diet_Type': user_input_dict['Diet_Type'],
                    'Activity_Level': user_input_dict['Activity_Level'],
                    'Meal_Type': dish['Meal_Type'],
                    'Calories': dish['Calories'],
                    'Protein': dish['Protein'],
                    'Carbs': dish['Carbs'],
                    'Fat': dish['Fat'],
                    'Fiber': dish['Fiber'],
                    'Dish_Diet_Type': dish['Diet_Type']
                }
                rows.append((row, dish['Dish_Name']))

            if not rows:
                final_recommendations[meal.lower()] = ["No suitable dishes found"]
                continue

            df_rows = pd.DataFrame([r[0] for r in rows])[all_columns]
            dish_names = [r[1] for r in rows]

            # Separate features
            X_num = df_rows[numerical_columns].values
            X_cat = encoder.transform(df_rows[categorical_columns])
            X_encoded = np.hstack([X_num, X_cat])

            # Predict and get top 5
            probs = model.predict_proba(X_encoded)[:, 1]
            top_indices = probs.argsort()[-5:][::-1]
            top_dishes = [dish_names[i] for i in top_indices]

            final_recommendations[meal.lower()] = top_dishes

        return jsonify({"recommendations": final_recommendations})

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
