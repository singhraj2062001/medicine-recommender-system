from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import pickle
import numpy as np
import pandas as pd


new_df = pd.read_pickle('new_df.pkl')
use_compos_df = pd.read_pickle('use_compos_df.pkl')

cv = CountVectorizer(max_features = 5000, stop_words='english')


app = Flask(__name__)


@app.route('/')
def index():
    with open('Medicine.jpg', 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return render_template('index.html' ,
                           base64_image=base64_image
                        )

@app.route('/recommend1')
def recommend_ui():
    return render_template('recommend1.html',
                           medicine_name=list(new_df['Name'].values)
                           )

@app.route("/recommend_medicines", methods=['POST'])
def recommend():
    vector  = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vector)[:, :2500]
    user_input = request.form.get('user_input')
    recommended_medicine = []
    med_index = new_df[new_df['Name'] == user_input].index[0]
    distances = similarity[med_index]
    med_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:13]
    for i in med_list:
        recommended_medicine.append(
            [new_df.iloc[i[0]].Name, new_df.iloc[i[0]].Score, new_df.iloc[i[0]].img_url, use_compos_df.iloc[i[0]].Composition,
             new_df.iloc[i[0]].Manufacturer, use_compos_df.iloc[i[0]].Uses])
    recommendations = sorted(recommended_medicine, reverse=True, key=lambda x: x[1])
    unique_data = {}
    for item in recommendations:
        medicine_name = item[0]
        if medicine_name not in unique_data:
            unique_data[medicine_name] = item
    # Extract the unique items from the dictionary
    unique_list = list(unique_data.values())
    # Printing the unique data
    del similarity
    del vector
    return render_template('recommend1.html', data=unique_list)



@app.route('/recommend2')
def recommend_ui_2():
    return render_template('recommend2.html',
                           med_compos=list(use_compos_df['Composition'].values)
                           )

@app.route("/recommend_medicines_2", methods=['POST'])
def recommend_2():
    user_input1 = request.form.get('user_input1')
    idx1 = use_compos_df.index[use_compos_df['Composition'] == user_input1]
    compos_input =  new_df['Composition'][idx1[0]]
    recommendations = []
    for index, row in new_df.iterrows():
        composition = row['Composition']
        if isinstance(composition, list):
            if all(query.lower() in ' '.join(composition).lower() for query in compos_input):
                med_name = row['Name']
                idx = use_compos_df.index[use_compos_df['Medicine Name'] == med_name]
                med_compos = use_compos_df['Composition'][idx[0]]
                med_uses = use_compos_df['Uses'][idx[0]]
                manufacturer = new_df['Manufacturer'][idx[0]]
                recommendations.append([row['Name'], row['Score'], manufacturer, row['img_url'], med_compos, med_uses])

    if recommendations:
        medicine_list = sorted(recommendations, reverse=True, key=lambda x: x[1])[:50]
        unique_data1 = {}

        for item in medicine_list:
            medicine_name = item[0]
            if medicine_name not in unique_data1:
                unique_data1[medicine_name] = item

        # Extract the unique items from the dictionary
        unique_list1 = list(unique_data1.values())

        # Printing the unique data
        return render_template('recommend2.html', data2 = unique_list1[0:12])

    else:
        return "No matching medicines found."






@app.route('/recommend3')
def recommend_ui_3():
    return render_template('recommend3.html',
                           med_uses=list(use_compos_df['Uses'].values)
                           )


@app.route("/recommend_medicines_3", methods=['POST'])
def recommend_3():
    user_input2 = request.form.get('user_input2')
    idx1 = use_compos_df.index[use_compos_df['Uses'] == user_input2]
    uses_inputs =  new_df['Uses'][idx1[0]]
    recommendations = []
    for index, row in new_df.iterrows():
        use = row['Uses']
        if isinstance(use, list):
            if all(query.lower() in ' '.join(use).lower() for query in uses_inputs):
                med_name = row['Name']
                index = use_compos_df.index[use_compos_df['Medicine Name'] == med_name]
                med_compos = use_compos_df['Composition'][index[0]]
                med_uses = use_compos_df['Uses'][index[0]]
                manufacturer = new_df['Manufacturer'][index[0]]
                recommendations.append([row['Name'], row['Score'], manufacturer, row['img_url'], med_compos, med_uses])

    if recommendations:
        medicine_list = sorted(recommendations, reverse=True, key=lambda x: x[1])[:50]
        unique_data = {}

        for item in medicine_list:
            medicine_name = item[0]
            if medicine_name not in unique_data:
                unique_data[medicine_name] = item

        unique_list2 = list(unique_data.values())

        return render_template('recommend3.html', data3 = unique_list2[0:12])
    else:
        return "No medicines found for this disease."






if __name__ == '__main__':
    app.run(debug = True)

