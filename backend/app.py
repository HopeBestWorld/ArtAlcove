import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

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

    matches = merged_df.groupby(
        ['product', 'siteurl', 'price', 'rating', 'imgurl', 'descr']
    ).agg({
        'review_title': list,
        'review_desc': list
    }).reset_index()
    matches_filtered_json = matches.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/stretched_canvas")
def episodes_search():
    text = request.args.get("product")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)