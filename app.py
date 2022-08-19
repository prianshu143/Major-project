import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model0 = pickle.load(open('Major-project.pkl','rb'))
model1 = pickle.load(open('tweetreviewLogistic.pkl','rb')) 
model2 = pickle.load(open('tweetreviewdecision_tree.pkl', 'rb'))



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus-data_Major.csv')
corpus1=corpus['corpus'].tolist()
X = cv.fit_transform(corpus1).toarray()

@app.route('/')
def home():
  
    return render_template("index.html")
@app.route('/aboutme')
def about():
    return render_template("aboutme.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  text = (request.args.get('Model'))
  text = [text]
  input_data = cv.transform(text).toarray()

  Model = (request.args.get('Model'))
  if Model=="Naive Bayes Algorithm":
    input_pred = model0.predict(X)
    #input_pred = input_pred.astype(int)
    print(input_pred)

  elif Model=="Logistic Regression Algorithm":
    input_pred = model1.predict(X)
    #input_pred = input_pred.astype(int)
    print(input_pred)

  else:
      input_pred = model2.predict(X)
      #input_pred = input_pred.astype(int)
      print(input_pred)

  if input_pred[0]==1:
    result= "User_verification is True"
  else:
    result="User_verification is False"
        
  return render_template('index.html', prediction_text='NLP Model  has predicted about the text : {}'.format(result))


if __name__=="__main__":
  app.run(debug=True)
