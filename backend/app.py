import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import re

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
art_csv_path = os.path.join(current_directory, 'Art.csv')  # CSV file
art_df = pd.read_csv(art_csv_path)

# Remove rows with any NaN values
art_df = art_df.dropna()

# Convert DataFrame to JSON
json_data = art_df.to_dict(orient="records")

# Save the JSON data to a file 
with open("Art.json", "w") as f:
    json.dump(json_data, f, indent=4)

json_file_path = os.path.join(current_directory, 'Art.json')  # JSON file

with open(json_file_path, 'r') as file:
    art_data = json.load(file)
    art_df = pd.DataFrame(art_data)

app = Flask(__name__)
CORS(app)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)