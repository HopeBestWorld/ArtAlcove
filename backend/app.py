import json
import os
import re
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
    colored_paper_df = pd.DataFrame(data['colored_paper'])
    colored_paper_reviews_df = pd.DataFrame(data['colored_paper_reviews'])
    fountain_pen_df = pd.DataFrame(data['fountain_pen'])
    fountain_pen_reviews_df = pd.DataFrame(data['fountain_pen_reviews'])
    gel_pen_df = pd.DataFrame(data['gel_pen'])
    gel_pen_reviews_df = pd.DataFrame(data['gel_pen_reviews'])
    markers_df = pd.DataFrame(data['markers'])
    markers_reviews_df = pd.DataFrame(data['markers_reviews'])
    oil_brush_df = pd.DataFrame(data['oil_brush'])
    oil_brush_reviews_df = pd.DataFrame(data['oil_brush_reviews'])
    oil_paint_df = pd.DataFrame(data['oil_paint'])
    oil_paint_reviews_df = pd.DataFrame(data['oil_paint_reviews'])
    sketchbooks_df = pd.DataFrame(data['sketchbooks'])
    sketchbooks_reviews_df = pd.DataFrame(data['sketchbooks_reviews'])
    watercolors_df = pd.DataFrame(data['watercolors'])
    watercolors_reviews_df = pd.DataFrame(data['watercolors_reviews'])
    watercolor_pads_df = pd.DataFrame(data['watercolor_pads'])
    watercolor_pads_reviews_df = pd.DataFrame(data['watercolor_pads_reviews'])
    watercolor_brushes_df = pd.DataFrame(data['watercolor_brushes'])
    watercolor_brushes_reviews_df = pd.DataFrame(data['watercolor_brushes_reviews'])
    watercolor_paper_df = pd.DataFrame(data['watercolor_paper'])
    watercolor_paper_reviews_df = pd.DataFrame(data['watercolor_paper_reviews'])
    watercolor_paper_df = pd.DataFrame(data['erasers'])
    watercolor_paper_reviews_df = pd.DataFrame(data['erasers_reviews'])

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
    merged_df2 = pd.merge(colored_paper_df, colored_paper_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(fountain_pen_df, fountain_pen_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(gel_pen_df, gel_pen_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(markers_df, markers_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(oil_brush_df, oil_brush_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(oil_paint_df, oil_paint_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(sketchbooks_df, sketchbooks_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(watercolors_df, watercolors_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(watercolor_pads_df, watercolor_pads_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(watercolor_brushes_df, watercolor_brushes_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(watercolor_paper_df, watercolor_paper_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(erasers_df, erasers_reviews_df, left_on='product', right_on='product', how='inner')
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

def filter_price(query, results):
    new_results = results
    num = re.findall(r'\$\d+\.?\d*|\d+\.?\d*\s*dollars', query)
    if 'cheap' in query:
        new_results = [result for result in results if result['price'] <= 20]
    elif 'expensive' in query:
        new_results = [result for result in results if result['price'] >= 50]
    if len(num) > 0:
        nums = [re.sub(r'^\$| dollars$', '', n) for n in num]
        new_results = [result for result in results if result['price'] <= int(nums[0])]
    return new_results

def get_new_price(results, isSet):
    for result in results:
        if isinstance(result['price'], str):
            max_price = float(result['price'].replace('$', '').strip())
        else:
            max_price = result['price']
        min_price = re.search(r"\$?(\d+\.?\d*)", result['price_range'])
        if min_price:
            min_price = float(min_price.group(1))
        else:
            min_price = max_price
    
        if isSet:
            result['price'] = min(min_price * 6, max_price)
        else:
            result['price'] = min_price
        
    return results

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search_cosine")
def search_cosine():
    query = request.args.get("query")
    toggle_mode = request.args.get("toggle", "unit")
    isSet = False
    if toggle_mode == "set":
        isSet = True

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
        pd.merge(calligraphy_df, calligraphy_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(colored_paper_df, colored_paper_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(fountain_pen_df, fountain_pen_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(gel_pen_df, gel_pen_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(markers_df, markers_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(oil_brush_df, oil_brush_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(oil_paint_df, oil_paint_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(sketchbooks_df, sketchbooks_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(watercolors_df, watercolors_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(watercolor_pads_df, watercolor_pads_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(watercolor_brushes_df, watercolor_brushes_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(watercolor_paper_df, watercolor_paper_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(erasers_df, erasers_reviews_df, left_on='product', right_on='product', how='inner')
    ])

    merged_df = merged_df.groupby(
        ['product', 'siteurl', 'price', 'rating', 'imgurl', 'descr', 'price_range']
    ).agg({
        'review_title': list,
        'review_desc': list
    }).reset_index()

    merged_df = merged_df.drop_duplicates(subset='product').reset_index(drop=True) #reset index
    merged_df[['review_title', 'review_desc']] = merged_df.apply(remove_duplicate_reviews, axis=1)

    vectorizer = build_vectorizer(max_features=1000, stop_words='english')
    
    merged_df['review_title_str'] = merged_df['review_title'].apply(lambda x: ' '.join(x))
    merged_df['review_desc_str'] = merged_df['review_desc'].apply(lambda x: ' '.join(x))
    merged_df['combined'] = merged_df['descr'] + " " + merged_df['product'] + " " + merged_df['review_title_str'] + " " + merged_df['review_desc_str']
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
                'price_range' : row['price_range'],
                'rating': row['rating'],
                'imgurl': row['imgurl'],
                'descr': row['descr'],
                'review_title': row['review_title'],
                'review_desc': row['review_desc'],
                'similarity': similarity
            })
    results = get_new_price(results, isSet)

    results.sort(key=lambda x: x['similarity'], reverse=True)
    results = filter_price(query, results)
    return json.dumps(results)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)