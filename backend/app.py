import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from analysis_a5 import build_vectorizer, get_sim
import numpy as np
from numpy import linalg as LA
from tfidf import query

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    stretched_canvas_df = pd.DataFrame(data['stretched_canvas'])
    stretched_canvas_reviews_df = pd.DataFrame(data['stretched_canvas_reviews'])
    alcohol_markers_df = pd.DataFrame(data['alcohol_markers'])
    alcohol_markers_reviews_df = pd.DataFrame(data['alcohol_markers_reviews'])
    colored_pencils_df = pd.DataFrame(data['colored_pencils'])
    colored_pencils_reviews_df = pd.DataFrame(data['colored_pencils_reviews'])
    drawing_pencils_df = pd.DataFrame(data['drawing_pencils'])
    drawing_pencils_reviews_df = pd.DataFrame(data['drawing_pencils_reviews'])
    graphite_df = pd.DataFrame(data['graphite'])
    graphite_reviews_df = pd.DataFrame(data['graphite_reviews'])
    oil_pastels_df = pd.DataFrame(data['oil_pastels'])
    oil_pastels_reviews_df = pd.DataFrame(data['oil_pastels_reviews'])
    pastel_pencils_df = pd.DataFrame(data['pastel_pencils'])
    pastel_pencils_reviews_df = pd.DataFrame(data['pastel_pencils_reviews'])
    soft_pastels_df = pd.DataFrame(data['soft_pastels'])
    soft_pastels_reviews_df = pd.DataFrame(data['soft_pastels_reviews'])
    acrylics_df = pd.DataFrame(data['acrylics'])
    acrylics_reviews_df = pd.DataFrame(data['acrylics_reviews'])
    acrylic_paintbrushes_df = pd.DataFrame(data['acrylic_paintbrushes'])
    acrylic_paintbrushes_reviews_df = pd.DataFrame(data['acrylic_paintbrushes_reviews'])
    erasers_df = pd.DataFrame(data['erasers'])
    erasers_reviews_df = pd.DataFrame(data['erasers_reviews'])
    calligraphy_df = pd.DataFrame(data['calligraphy'])
    calligraphy_reviews_df = pd.DataFrame(data['calligraphy_reviews'])

app = Flask(__name__)
CORS(app)

# # Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(stretched_canvas_df, stretched_canvas_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df2 = pd.merge(alcohol_markers_df, alcohol_markers_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(colored_pencils_df, colored_pencils_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(drawing_pencils_df, drawing_pencils_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(graphite_df, graphite_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(oil_pastels_df, oil_pastels_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(pastel_pencils_df, pastel_pencils_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(soft_pastels_df, soft_pastels_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(acrylics_df, acrylics_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(acrylic_paintbrushes_df, acrylic_paintbrushes_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(calligraphy_df, calligraphy_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    

    matches = merged_df.groupby(
        ['product', 'siteurl', 'price', 'rating', 'imgurl', 'descr']
    ).agg({
        'review_title': list,
        'review_desc': list
    }).reset_index()
    matches_filtered_json = matches.to_json(orient='records')
    return matches_filtered_json

def remove_duplicate_reviews(row):
    combined_reviews = list(zip(row['review_title'], row['review_desc']))
    unique_reviews = list(set(combined_reviews))
    if unique_reviews:
        unique_titles, unique_descs = zip(*unique_reviews)
        return pd.Series([list(unique_titles), list(unique_descs)])
    else:
        return pd.Series([[], []])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search_cosine")
def search_cosine():
    query = request.args.get("query")
    if not query:
        return json.dumps([])

    merged_df = pd.concat([
        pd.merge(stretched_canvas_df, stretched_canvas_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(alcohol_markers_df, alcohol_markers_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(colored_pencils_df, colored_pencils_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(drawing_pencils_df, drawing_pencils_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(graphite_df, graphite_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(oil_pastels_df, oil_pastels_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(pastel_pencils_df, pastel_pencils_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(soft_pastels_df, soft_pastels_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(acrylics_df, acrylics_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(acrylic_paintbrushes_df, acrylic_paintbrushes_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(calligraphy_df, calligraphy_reviews_df, left_on='product', right_on='product', how='inner')
    ])

    merged_df = merged_df.groupby(
        ['product', 'siteurl', 'price', 'rating', 'imgurl', 'descr']
    ).agg({
        'review_title': list,
        'review_desc': list
    }).reset_index()

    merged_df = merged_df.drop_duplicates(subset='product').reset_index(drop=True) #reset index
    merged_df[['review_title', 'review_desc']] = merged_df.apply(remove_duplicate_reviews, axis=1)

    vectorizer = build_vectorizer(max_features=1000, stop_words='english')
    
    merged_df['combined'] = merged_df['descr'] + " " + merged_df['product']
    tfidf_matrix = vectorizer.fit_transform(merged_df['combined'])
    query_vector = vectorizer.transform([query])

    results = []
    for index, product_vector in enumerate(tfidf_matrix): #Iterate tfidf matrix directly
        row = merged_df.iloc[index]
        similarity = get_sim(query_vector, product_vector)
        if similarity > 0:
            results.append({
                'product': row['product'],
                'siteurl': row['siteurl'],
                'price': row['price'],
                'rating': row['rating'],
                'imgurl': row['imgurl'],
                'descr': row['descr'],
                'review_title': row['review_title'],
                'review_desc': row['review_desc'],
                'similarity': similarity
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return json.dumps(results)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)