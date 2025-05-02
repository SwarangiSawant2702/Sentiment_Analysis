from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification

app = Flask(__name__)

# 🧠 Load your trained model and tokenizer

tokenizer = AlbertTokenizer.from_pretrained("saved_model/tokenizer")
model = AlbertForSequenceClassification.from_pretrained("saved_model")
model.eval()

# 🔍 Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/review", methods=['POST'])
def review():
    if request.method == 'POST':
        try:
            # Get product URL from form
            product_link = request.form['content']

            # Setup headless Selenium
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

            # Load product page
            driver.get(product_link)
            product_html = bs(driver.page_source, 'html.parser')

            # Extract product name
            product_name = product_html.findAll("div", {"class": "C7fEHH"})[0].div.text

            # Extract all reviews
            comment_boxes = product_html.findAll("div", {"class": "RcXBOT"})

            reviews = []
            for comment_box in comment_boxes:
                try:
                    name = comment_box.div.div.find_all('p', {'class': '_2NsDsF AwS1CA'})[0].text
                except:
                    name = 'No Name'

                try:
                    rating = comment_box.div.div.div.div.text
                except:
                    rating = 'No Rating'

                try:
                    comment_head = comment_box.div.div.div.p.text
                except:
                    comment_head = 'No Comment Heading'

                try:
                    comment = comment_box.find('div', {'class': 'ZmyHeo'}).text.replace('READ MORE', '').strip()
                except:
                    comment = 'No Comment'

                # 🧠 Predict sentiment
                full_text = f"{comment_head} {comment}"
                sentiment = predict_sentiment(full_text)

                review_dict = {
                    "Name": name,
                    "Rating": rating,
                    "CommentHead": comment_head,
                    "Comment": comment,
                    "Sentiment": sentiment
                }
                reviews.append(review_dict)

            driver.quit()

            return render_template('results.html', product_name=product_name, reviews=reviews)

        except Exception as e:
            print('The Exception message is:', e)
            return render_template('results.html', product_name=None, reviews=[], error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
