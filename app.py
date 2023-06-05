from flask import Flask, render_template, url_for, request, jsonify
from predict import predictTags, dump2dict

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        predictions = predictTags(text)
        results = dump2dict(predictions, text)
        # return render_template('results.html', results = results)
        return jsonify({"entities":results})


if __name__ == '__main__':
	app.run(debug=True)