#Author: Patrick Tan
#contact: patrick.patricktan@gmail.com
#This will take in the records from the survey and generate a feedback base on GPT-2 model

from flask import Flask, render_template, request
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Load survey dataset
survey_data = pd.read_csv('survey.csv')
unique_french_toasts = survey_data['French Toast'].unique().astype(str)

# Load recipes dataset
recipes_data = pd.read_csv('recipes.csv')

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model to evaluation mode
model = model.to(device)
model.eval()


@app.route('/')
def index():
    return render_template('feedback.html', result=None, matched_records=None)


@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        feedback = request.form['feedback']
        # Tokenize input text
        input_ids = tokenizer.encode(feedback, return_tensors='pt').to(device)
        # Generate text based on input text
        output = model.generate(input_ids, max_length=500, temperature=0.7, num_return_sequences=1)
        # Decode generated text
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Perform sentiment analysis (Placeholder - you can implement your own sentiment analysis logic here)
        positive_keywords = ["delicious", "amazing", "great", "wonderful", "tasty","good","fantastic","impressive","awesome"]  # Add more positive keywords as needed
        if any(keyword in output_text for keyword in positive_keywords):
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        # Extract the highest possible French Toast code from the generated text
        max_french_toast = None
        max_probability = -1
        for french_toast in unique_french_toasts:
            probability = str(output_text).count(french_toast)
            if probability > max_probability:
                max_french_toast = french_toast
                max_probability = probability

        # Search for the matched product code from recipes.csv
        matched_records = None
        if max_french_toast:
            matched_records = recipes_data[recipes_data['product_code'] == max_french_toast].to_dict('records')

        result = f"Sentiment: {sentiment}\nHighest possible French Toast code: {max_french_toast}\nGenerated text: {output_text}"
        return render_template('feedback.html', result=result, matched_records=matched_records)


if __name__ == '__main__':
    app.run(debug=True)
