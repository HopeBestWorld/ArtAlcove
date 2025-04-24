from collections import defaultdict
import json
import os
import re
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from analysis import build_vectorizer, get_sim
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import TruncatedSVD


from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

from analysis import build_vectorizer, get_sim, edit_distance
from tfidf import query

# ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

PRODUCT_WEIGHT = 4
DESCR_WEIGHT = 3
REVIEW_TITLE_WEIGHT = 2
REVIEW_DESC_WEIGHT = 1

def remove_duplicate_reviews(row):
    combined_reviews = list(zip(row['review_title'], row['review_desc']))
    filtered_reviews = [(title, desc) for title, desc in combined_reviews if title.strip() and desc.strip()]
    unique_reviews = list(set(filtered_reviews))
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

def multiple_categories(results):
    lowest = []
    remaining = []
    top = []
    category_groups = defaultdict(list)
    for result in results:
        category = result['category_x']
        similarity = result['similarity']
        if similarity >= 0.25:
            category_groups[category].append(result)
        else:
            lowest.append(result)
    for cat in category_groups:
        top += category_groups[cat][:3]
        remaining += category_groups[cat][3:]
    return (top + remaining + lowest)[:20]

def insertion_cost(message, j):
    return 1


def deletion_cost(query, i):
    return 1


def substitution_cost(query, message, i, j):
    if query[i - 1] == message[j - 1]:
        return 0
    else:
        return 1

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
    erasers_df = pd.DataFrame(data['erasers'])
    erasers_reviews_df = pd.DataFrame(data['erasers_reviews'])
    clay_df = pd.DataFrame(data['clay'])
    clay_reviews_df = pd.DataFrame(data['clay_reviews'])
    ceramic_df = pd.DataFrame(data['ceramic'])
    ceramic_reviews_df = pd.DataFrame(data['ceramic_reviews'])
    beads_df = pd.DataFrame(data['beads'])
    beads_reviews_df = pd.DataFrame(data['beads_reviews'])
    weaving_df = pd.DataFrame(data['weaving'])
    weaving_reviews_df = pd.DataFrame(data['weaving_reviews'])
    yarn_df = pd.DataFrame(data['yarn'])
    yarn_reviews_df = pd.DataFrame(data['yarn_reviews'])
    knitting_df = pd.DataFrame(data['knitting'])
    knitting_reviews_df = pd.DataFrame(data['knitting_reviews'])
    pottery_df = pd.DataFrame(data['pottery'])
    pottery_reviews_df = pd.DataFrame(data['pottery_reviews'])
    wood_df = pd.DataFrame(data['wood'])
    wood_reviews_df = pd.DataFrame(data['wood_reviews'])
    felting_df = pd.DataFrame(data['felting'])
    felting_reviews_df = pd.DataFrame(data['felting_reviews'])
    

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
        pd.merge(erasers_df, erasers_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(clay_df, clay_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(ceramic_df, ceramic_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(beads_df, beads_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(weaving_df, weaving_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(yarn_df, yarn_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(knitting_df, knitting_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(pottery_df, pottery_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(wood_df, wood_reviews_df, left_on='product', right_on='product', how='inner'),
        pd.merge(felting_df, felting_reviews_df, left_on='product', right_on='product', how='inner')



    ])

    merged_df = merged_df.groupby(
        ['product', 'siteurl', 'price', 'rating', 'imgurl', 'descr', 'price_range', 'category_x']
    ).agg({
        'review_title': list,
        'review_desc': list
    }).reset_index()

    merged_df = merged_df.drop_duplicates(subset='product').reset_index(drop=True)
    merged_df[['review_title', 'review_desc']] = merged_df.apply(remove_duplicate_reviews, axis=1)

    vectorizer = build_vectorizer(max_features=1000, stop_words='english')

    merged_df['review_title_str'] = merged_df['review_title'].apply(lambda x: ' '.join(x))
    merged_df['review_desc_str'] = merged_df['review_desc'].apply(lambda x: ' '.join(x))
    merged_df['combined'] = merged_df['descr'] * DESCR_WEIGHT + " " + merged_df['product'] * PRODUCT_WEIGHT + " " + merged_df['review_title_str'] * REVIEW_TITLE_WEIGHT + " " + merged_df['review_desc_str'] * REVIEW_DESC_WEIGHT    
    tfidf_matrix = vectorizer.fit_transform(merged_df['combined'])

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
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
    merged_df2 = pd.merge(clay_df, clay_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(ceramic_df, ceramic_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(beads_df, beads_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(weaving_df, weaving_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(yarn_df, yarn_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(knitting_df, knitting_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(pottery_df, pottery_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(wood_df, wood_reviews_df, left_on='product', right_on='product', how='inner')
    merged_df = pd.concat([merged_df, merged_df2])
    merged_df2 = pd.merge(felting_df, felting_reviews_df, left_on='product', right_on='product', how='inner')
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
    return render_template('base.html', title="sample html")

@app.route("/search_cosine")
def search_cosine():
    query_str = request.args.get("query")
    toggle_mode = request.args.get("toggle", "unit")
    isSet = False
    if toggle_mode == "set":
        isSet = True

    if not query_str:
        return json.dumps({"results": [], "query_latent_topics": [], "suggestions": []})

    query_vector = vectorizer.transform([query_str])

    # Apply SVD with a smaller number of components for interpretability
    n_components_explain = 20
    svd_explain = TruncatedSVD(n_components=n_components_explain)
    tfidf_matrix_reduced_explain = svd_explain.fit_transform(tfidf_matrix)
    query_vector_reduced_explain = svd_explain.transform(query_vector)

    # Get the vocabulary
    feature_names = vectorizer.get_feature_names_out()
    components = svd_explain.components_

    def get_top_words(component, n=5):
        """Gets the top n words for a given SVD component."""
        sorted_word_indices = np.argsort(np.abs(component))[::-1]
        return [feature_names[index] for index in sorted_word_indices[:n]]

    results = []

    # Find the top 3 most influential latent dimensions for this query
    query_dimension_strengths = np.abs(query_vector_reduced_explain[0])
    query_top_dimension_indices = np.argsort(query_dimension_strengths)[::-1][:3]

    query_latent_topics = []
    for index in query_top_dimension_indices:
        query_latent_topics += (get_top_words(components[index]))
    query_latent_topics = list(set(query_latent_topics))

    for index, product_vector_reduced in enumerate(tfidf_matrix_reduced_explain):
        row = merged_df.iloc[index]
        similarity = max(0, 0, get_sim(query_vector_reduced_explain, product_vector_reduced))

        # Find the top 3 most influential latent dimensions for this product
        product_dimension_strengths = np.abs(product_vector_reduced)
        top_dimension_indices = np.argsort(product_dimension_strengths)[::-1][:3]

        product_latent_topics = []
        for idx in top_dimension_indices:
            product_latent_topics += (get_top_words(components[idx]))
        product_latent_topics = list(set(product_latent_topics))

        query_lower = query_str.lower()
        product_name = row['product'].lower()
        if product_name in query_lower:
            results.append({
                'product': row['product'],
                'siteurl': row['siteurl'],
                'price': row['price'],
                'price_range': row['price_range'],
                'rating': row['rating'],
                'imgurl': row['imgurl'],
                'descr': row['descr'],
                'review_title': row['review_title'],
                'review_desc': row['review_desc'],
                'category_x': row['category_x'],
                'similarity': 1.0,
                'latent_topics': product_latent_topics
            })
        elif similarity > 0:
            results.append({
                'product': row['product'],
                'siteurl': row['siteurl'],
                'price': row['price'],
                'price_range': row['price_range'],
                'rating': row['rating'],
                'imgurl': row['imgurl'],
                'descr': row['descr'],
                'review_title': row['review_title'],
                'review_desc': row['review_desc'],
                'category_x': row['category_x'],
                'similarity': float(similarity.item()),
                'latent_topics': product_latent_topics
            })

    results = get_new_price(results, isSet)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    results = filter_price(query_str, results)
    results = multiple_categories(results)

    suggestions = []
    if not results:
        all_products = merged_df['product'].unique().tolist()
        suggestion_distances = []
        for product in all_products:
            distance = edit_distance(query_str, product, insertion_cost, deletion_cost, substitution_cost)
            suggestion_distances.append((distance, product))

        suggestion_distances.sort(key=lambda item: item[0])
        suggestions = [product for distance, product in suggestion_distances[:5]] # Get top 5 suggestions

    return json.dumps({"results": results, "query_latent_topics": query_latent_topics, "suggestions": suggestions})

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5001)