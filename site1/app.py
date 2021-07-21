import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path, convert_from_bytes


#UPLOAD_FOLDER = '/home/appuser/mla/app/uploads/'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
#ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'
#app.config['UPLOAD_PATH'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    extracted = { "author": "mark twiain",
      "title": "tom sawyer",}
    return render_template('index.html', extracted=extracted)
    #return render_template('index.html')

@app.route('/', methods=["POST"])
def upload_file():
    
    if request.method == 'POST':
        
        # check if the post request has the file part
        if 'file' not in request.files:
            print("no hay archivo")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #print(extracted["author"])
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
            file.save(filepath)
            
            imagepath = os.path.splitext(filepath)[0]+".jpg"
            
            pages = convert_from_path(filepath, single_file=True, size=(400, None))
            firstpage = pages[0]
            firstpage.save(imagepath, 'JPEG')            
            
            
            print("saved")
            #return redirect(url_for('download_file', name=filename))
            
    
    extracted = extract_metadata(filepath)
    extracted['preview'] = imagepath
            
    return render_template('index_results.html', extracted=extracted)


@app.route('/predict', methods=["GET", "POST"])
def predict():
    # Not yet implement
    clicked=None
    if request.method == "POST":
        clicked = request.json['file']
    return render_template('index.html')


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def extract_metadata(pdffile_path):
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



"""
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'
"""


"""
def upload_file():
    output = {}

    if request.method == 'POST':
        if 'file' not in request.files:
            output["flash"] = "Please upload a PDF document"
            return Response(json.dumps(output), mimetype='text/json')
        
        file = request.files['file']
        print(file.filename)
        if file.filename == '':
            output["flash"] = "Please upload a PDF document"
            return Response(json.dumps(output), mimetype='text/json')
        if not allowed_file(file.filename):
            output["flash"] = 'Wrong file format. Please upload a PDF document!'
            return Response(json.dumps(output), mimetype='text/json')
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            file_saved = False
            waited = 0
            while not file_saved:
                if not os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                    time.sleep(1)
                    waited += 1
                    print("waiting")
                else:
                    file_saved = True
                if waited > 20:
                    return redirect(request.host_url)
            for i in range(3):
                while True:
                    try:
                        v, metadata = pubmex.predict(app.config['UPLOAD_FOLDER'] + filename)
                        img = Image.fromarray(v.get_image()[:, :, ::-1])
                        img_path = app.config['UPLOAD_FOLDER'] + filename[:-4] + ".jpg"
                        img.save(img_path)
                        output["output"] = metadata
                        output["image_path"] = app.config['UPLOAD_FOLDER'] + filename[:-4] + ".jpg"
                        return Response(json.dumps(output), mimetype='text/json')
                    except:
                        output["flash"] = "Something went wrong uploading the file - please try again."
                        if i == 2:
                            return Response(json.dumps(output), mimetype='text/json')
                        continue
                    break
    return render_template("index.html")
"""
