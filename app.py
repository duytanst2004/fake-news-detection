from flask import Flask, request, render_template
from predict import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            predictions = predict(text)
            return render_template('index.html', text=text, predictions=predictions)
        return render_template('index.html', error="Please enter some text.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)