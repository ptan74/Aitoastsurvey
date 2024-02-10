#generate random data record for test
#无需用户输入，自动生成测试数据记录

from flask import Flask
import csv
import random
import string

app = Flask(__name__)


@app.route('/')
def generate_data():
    # Define the records
    records = [
        ['FT001', 2, 1, 2, 'The bread looks cheap. Display is bad.',
         'One bite, the bread falls apart. The skin is soft and flaky. Inner and outer both failed. Not delicious as well.'],
        ['FT001', 4, 4, 4, 'The bread looks hard and crispy. It is very presentable.',
         'I love it with the first bite. The bread is hard. Its crispy skin and sturdy bread bonding made it to a first class toast.']
    ]

    # Generate additional random data to reach 100 rows
    for _ in range(98):
        # Generate random values
        texture = random.randint(1, 5)
        crispiness = random.randint(1, 5)
        taste = random.randint(1, 5)
        visual = ''.join(random.choices(string.ascii_letters, k=20))  # Random visual description
        satisfaction = ''.join(random.choices(string.ascii_letters, k=50))  # Random satisfaction feedback
        french_toast = 'FT001'  # Default value

        # Append the record to the records list
        records.append([french_toast, texture, crispiness, taste, visual, satisfaction])

    # Write data to CSV file
    with open('survey.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['French Toast', 'Texture', 'Crispiness', 'Taste', 'Visual Presentation', 'Overall Satisfaction'])
        writer.writerows(records)

    return 'Data generated successfully!'


if __name__ == '__main__':
    app.run(debug=True)
