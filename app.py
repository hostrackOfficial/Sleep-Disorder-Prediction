from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("C:/Users/hostr/OneDrive/Documents/Projects/Website/Working/Sleep Disorder Prediction/nishant-disorder.csv")
df = df.dropna()

#label encoding
label_encode = {"Sleep Disorder": {"None": 1, "Sleep Apnea": 2, "Insomnia": 3}}

# data preparation
X = df[['Stress Level', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Age', 'Sleep Duration']]
Y = df['Sleep Disorder']

# Create and train KNN classifier
KNN = KNeighborsClassifier()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
KNN.fit(x_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def predict_sleep_disorder():
    result = None  
    if request.method == 'POST':
        subs = ['Stress Level', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Age', 'Sleep Duration']
        dicti = {}
        for i in subs:
            dicti[i] = float(request.form.get(i))

        test = pd.DataFrame({
            'Stress Level': [dicti['Stress Level']],
            'Physical Activity Level': [dicti['Physical Activity Level']],
            'Heart Rate': [dicti['Heart Rate']],
            'Daily Steps': [dicti['Daily Steps']],
            'Age': [dicti['Age']],
            'Sleep Duration': [dicti['Sleep Duration']]
        })

        test_pred = KNN.predict(test)

        result=test_pred

    return render_template('app.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
