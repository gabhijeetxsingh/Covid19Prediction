from flask import Flask, render_template
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/')
def hello_world():
    
    # Code for inference
    inputFeatures = [90, 0 , 25, 0, -1]
    infProb = clf.predict_proba([inputFeatures])[0][1]

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)    