import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path, convert_from_bytes
from pubmex.pubmexinference import PubMexInference
from cermine.main import CermineTextExtractor

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'PDF'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'

# Initialize pubmex
pubmex = PubMexInference(
    model_dump='pubmex/vision_model/model_final.pth', 
    config_file='pubmex/configs/train_config.yaml',
    use_cuda=False,
    )
# Initialize Cermine
textExtractor = CermineTextExtractor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/', methods=["POST"])
def upload_file():
    
    if request.method == 'POST':
        
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No PDF file selected!')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Wrong file format. Please upload a PDF document!')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
            file.save(filepath)
            
            imagepath = os.path.splitext(filepath)[0]+".jpg"
            
            pages = convert_from_path(filepath, single_file=True, size=(400, None))
            firstpage = pages[0]
            firstpage.save(imagepath, 'JPEG')            
            
            #return redirect(url_for('download_file', name=filename))
            
    textExtractor.cermine_extract('uploads',True, True)
    extracted = extract_metadata(filepath)
    extracted['preview'] = imagepath
            
    return render_template('index.html', extracted=extracted)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def extract_metadata(pdffile_path):
    
    # Get vision output
    # image_output = pubmex.alt_predict(pdffile_path) 

    #TODO      add text processing here 
    #:         [Low priority] make it pararel with text model (might use multitreading)
        
    #load model
    #model = pickle.load(open('model.pkl', 'rb'))
    
    #get features from pdf
    #final_features = get_features(pdffile_path)
    
    #make prediction
    #prediction = model.predict(final_features)
    
    #dummy data to send to the view template    
    extracted = {
      "author": "author",
      "title": "title",
      "affiliation": "affiliation",
      "date": "date",
      "journal": "journal",
      "address": "address",
      "abstract": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum",
      "doi": "DOI",
      "email": "email@aol.com"
      }
    
    return extracted
