import os
import flask
import joblib
import numpy as np

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Saved machine learning model load karein
model = joblib.load('skin_model.pkl')

@app.route('/')
def home():
    return flask.render_template('index.html', prediction=None, confidence=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in flask.request.files:
        return flask.redirect(flask.request.url)
    file = flask.request.files['file']
    if file.filename == '':
        return flask.redirect(flask.request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Model se prediction aur confidence score nikalna
        sample_features = np.random.rand(1, 10)
        prediction = model.predict(sample_features)[0]
        
        # Yahan aap apne model ke result ke mutabiq details set kar sakte hain
        disease_name = str(prediction)
        confidence_score = "92.5%"  # Example score
        description = "Yeh ek sample analysis hai. Kripya kisi doctor ya dermatologist se confirm zaroor karwayein."
        
        return flask.render_template('index.html', 
                                     prediction=disease_name, 
                                     confidence=confidence_score, 
                                     desc=description,
                                     image_file=file.filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)