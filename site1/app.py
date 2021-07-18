from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    # Not yet implement
    clicked=None
    if request.method == "POST":
        clicked = request.json['file']
    return render_template('index.html')

@app.route('/uploadpdf', methods=["GET", "POST"])
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
