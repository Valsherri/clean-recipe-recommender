
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from surprise import SVD
from surprise import Dataset, Reader

app = Flask(__name__)

# Load data
mergedsubset_df = pd.read_csv('mergedsubset.csv')
ratings_df = mergedsubset_df[['AuthorId_recipe', 'RecipeId_encoded', 'Rating']].dropna()

# Load model
with open('model.pkl', 'rb') as f:
    svd = pickle.load(f)

# Fallback recommendation logic
def fallback_recommendations(top_n=5):
    top_recipes = (
        mergedsubset_df.groupby('RecipeId_encoded')
        .agg({'Rating': ['mean', 'count']})
        .reset_index()
    )
    top_recipes.columns = ['RecipeId_encoded', 'AvgRating', 'RatingCount']
    top_recipes = top_recipes.sort_values(by=['RatingCount', 'AvgRating'], ascending=False)

    top_ids = top_recipes['RecipeId_encoded'].head(top_n)
    return mergedsubset_df[mergedsubset_df['RecipeId_encoded'].isin(top_ids)]['Name'].unique().tolist()

# Recommendation function
def recommend_for_user(user_id, top_n=5):
    user_id = int(user_id)
    all_recipes = mergedsubset_df['RecipeId_encoded'].unique()
    rated = ratings_df[ratings_df['AuthorId_recipe'] == user_id]['RecipeId_encoded'].tolist()

    if not rated:
        return fallback_recommendations(top_n)

    candidates = [rid for rid in all_recipes if rid not in rated]
    predictions = [(rid, svd.predict(user_id, rid).est) for rid in candidates]
    top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    recommended = []
    for rid, _ in top_preds:
        name = mergedsubset_df.loc[mergedsubset_df['RecipeId_encoded'] == rid, 'Name'].values[0]
        recommended.append(name)
    return recommended

# Flask route
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    recommendations = recommend_for_user(user_id)
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    from os import environ
    port = int(environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)

