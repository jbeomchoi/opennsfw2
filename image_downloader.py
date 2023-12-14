import pandas as pd
import requests
import os
import sys
import opennsfw2 as n2

if len(sys.argv) != 4:
    print("python3 image_downloader.py <csv input path> <path to download image> <csv output path>")
    sys.exit(1)

csv_input = sys.argv[1]
target_folder = sys.argv[2]
csv_output_path = sys.argv[3]

df = pd.read_csv(csv_input)

scores = []

for index, row in df.iterrows():
    url = row['url']
    class_name = row['className']

    class_path = os.path.join(target_folder, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    save_path = os.path.join(class_path, url.replace('/', '|'))

    res = requests.get(url)
    if res.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(res.content)

        score = n2.predict_image(save_path)
        scores.append(score)

df['opennsfw2_score'] = scores

df.to_csv(csv_output_path, index=False)
