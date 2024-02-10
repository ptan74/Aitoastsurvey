#Author: Patrick Tan
#contact: patrick.patricktan@gmail.com
#This generates a french toast feedback form

from flask import Flask, render_template, request
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('survey.html')

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    if request.method == 'POST':
        texture = request.form['texture']
        crispiness = request.form['crispiness']
        taste = request.form['taste']
        visual = request.form['visual']
        satisfaction = request.form['satisfaction']

        # Default value for French Toast
        french_toast = 'FT001'

        # Append data to CSV file
        with open('survey.csv', mode='a', newline='') as file:
            if file.tell() == 0:
                # Write headers if the file is empty
                file.write('French Toast,Texture,Crispiness,Taste,Visual Presentation,Overall Satisfaction')

            # Append data to the next line
            file.write(f'\n{french_toast},{texture},{crispiness},{taste},{visual},{satisfaction}')

        return 'Survey submitted successfully!'

if __name__ == '__main__':
    app.run(debug=True)
