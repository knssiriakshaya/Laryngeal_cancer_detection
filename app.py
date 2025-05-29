from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from model_loader import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict the class
            predicted_class = predict_image(filepath)
            
            return render_template('result.html', image_path=filepath, result=predicted_class)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
