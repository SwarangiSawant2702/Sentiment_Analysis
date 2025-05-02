import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from collections import Counter
import re
import logging

app = Flask(__name__)

# 🧠 Load model & tokenizer
tokenizer = AlbertTokenizer.from_pretrained("saved_model/tokenizer")
model = AlbertForSequenceClassification.from_pretrained("saved_model")
model.eval()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# 🔍 Predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

# 🔎 Extract good/bad features dynamically
def extract_features(reviews):
    good_phrases = []
    bad_phrases = []

    for review in reviews:
        comment = review["Comment"].lower()
        sentiment = review["Sentiment"]

        words = re.findall(r'\w+', comment)
        bigrams = [" ".join(pair) for pair in zip(words, words[1:])]

        if sentiment == "Positive":
            good_phrases.extend(bigrams)
        elif sentiment == "Negative":
            bad_phrases.extend(bigrams)

    top_good = [phrase for phrase, _ in Counter(good_phrases).most_common(10)]
    top_bad = [phrase for phrase, _ in Counter(bad_phrases).most_common(10)]

    return top_good, top_bad

# 🔁 Append reviews to CSV if not already present
def append_to_csv(reviews, csv_file="reviews.csv"):
    # Load existing CSV or create a new one with headers if the file doesn't exist
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=["Name", "Rating", "CommentHead", "Comment", "Sentiment"])

    # Convert reviews to DataFrame
    df_reviews = pd.DataFrame(reviews)

    # Remove duplicates by comparing comments (to avoid adding the same review)
    df_combined = pd.concat([df_existing, df_reviews]).drop_duplicates(subset=["Comment"], keep='last')

    # Save to CSV
    df_combined.to_csv(csv_file, index=False)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/review", methods=['POST'])
def review():
    driver = None
    try:
        product_link = request.form['content']

        # Setup headless browser
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # Load the page
        driver.get(product_link)
        soup = bs(driver.page_source, 'html.parser')

        # Product Name
        product_name_divs = soup.find_all("div", {"class": "C7fEHH"})
        product_name = product_name_divs[0].div.text if product_name_divs else "Unknown Product"

        # Extract Reviews
        comment_boxes = soup.find_all("div", {"class": "RcXBOT"})
        reviews = []
        sentiment_counts = {"Positive": 0, "Negative": 0}

        for box in comment_boxes:
            try:
                name = box.div.div.find_all('p', {'class': '_2NsDsF AwS1CA'})[0].text
            except:
                name = 'No Name'

            try:
                rating = box.div.div.div.div.text
            except:
                rating = 'No Rating'

            try:
                comment_head = box.div.div.div.p.text
            except:
                comment_head = 'No Comment Heading'

            try:
                comment = box.find('div', {'class': 'ZmyHeo'}).text.replace('READ MORE', '').strip()
            except:
                comment = 'No Comment'

            full_text = f"{comment_head} {comment}"
            sentiment = predict_sentiment(full_text)
            sentiment_counts[sentiment] += 1

            reviews.append({
                "Name": name,
                "Rating": rating,
                "CommentHead": comment_head,
                "Comment": comment,
                "Sentiment": sentiment
            })

        # 🔍 Extract & clean features
        good_features, bad_features = extract_features(reviews)

        # Store reviews in CSV
        append_to_csv(reviews)

        total_reviews = len(reviews)
        positive_percentage = round((sentiment_counts["Positive"] / total_reviews) * 100, 1) if total_reviews else 0
        negative_percentage = round((sentiment_counts["Negative"] / total_reviews) * 100, 1) if total_reviews else 0

        return render_template('results.html',
                               product_name=product_name,
                               reviews=reviews,
                               sentiment_counts=sentiment_counts,
                               positive_percentage=positive_percentage,
                               negative_percentage=negative_percentage,
                               good_features=good_features,
                               bad_features=bad_features)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return render_template('results.html', product_name=None, reviews=[], error="An error occurred while processing your request.")

    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    app.run(debug=True)
