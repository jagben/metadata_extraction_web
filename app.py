import glob, os, os.path
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path, convert_from_bytes
from pubmex.pubmexinference import PubMexInference
from cermine.main import CermineTextExtractor
from pandas.core.algorithms import value_counts
from feature_extraction.layout_feature_extraction.main import FeatureExtractor
from feature_extraction.context_feature_extraction.main import ContextFeatureExtractor

from model.bi_lstm_submodel.Model1 import BiLSTM
from model.Model2 import BiLSTM2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import os

torch.manual_seed(10)
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'
file_name = ""

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
            filelist = glob.glob('./uploads/*')
            for f in filelist:
                os.remove(f)
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


def step1():
    textExtractor = CermineTextExtractor()
    # Will only output something if there is a new, unprocessed pdf in the folder
    # Also it doesn't work if there are spaces in foldernames in the path
    textExtractor.cermine_extract('uploads/',True, True)

def step2():
    # change path here to the output of cermine
    featureExtractor = FeatureExtractor('./uploads/')
    featureExtractor.get_dataset_features()

def step3():
    # Change path here
    contextExtractor = ContextFeatureExtractor("./word_lists/", "./features/")
    contextExtractor.get_features()
    

def step4(features):
    # 1041 embedding size
    print("~~~~~~~~~~~~~~~~~~~")
    model = BiLSTM(1041)
    model.load_state_dict(torch.load("./model/bi_lstm_submodel/model_fixed_v1.pt"))
    model.to(device)
    model.eval()
    output = model(features)
    output = F.softmax(output, dim=1)
    return output


def step5(features, i):
    model = BiLSTM2(20)
    model.load_state_dict(torch.load("./model/model_fixed_v1.pt"))
    model.to("cpu")
    model.eval()
    output = model(features)
    final_output = F.softmax(output, dim=1)
    
    new_out = final_output.detach().numpy()

    title = ''
    author = ''
    journal = ''
    aff = ''
    abstract = ''
    doi = ''
    addr = ''
    email = ''
    date = ''
    other = ''
    new_out = i
    for i, score in enumerate(new_out):
        word = score[0]
        cl = score[-1].argmax().item()
        if cl == 0:
            abstract+= word
        if cl == 1:
            addr+= word
        if cl == 2:
            aff+= word
        if cl == 3:
            author+= word
        if cl == 4:
            date+= word
        if cl == 5:
            doi+= word
        if cl == 6:
            email+= word
        if cl == 7:
            journal+= word
        if cl == 8:
            title+= word
        if cl == 9:
            other+= word

    return {
      "author": author,
      "title": title,
      "affiliation": aff,
      "date": date,
      "journal": journal,
      "address": addr,
      "abstract": abstract,
      "doi": doi,
      "email": email
      }

def extract_metadata(pdffile_path):
    
    # Get vision output
    image_output = pubmex.alt_predict(pdffile_path) 
    print(image_output)

    #TODO      add text processing here 
    #:         [Low priority] make it pararel with text model (might use multitreading)
    
    step1()
    step2()
    step3()
    words = []
    for file in os.scandir('./features/'):
        if file.name.endswith('csv'):
            vector = pd.read_csv(file)
            if (vector.shape[1] == 1044):
                words = vector.iloc[:,2]
                vector.drop([vector.columns[0], vector.columns[1], vector.columns[2]], axis=1, inplace=True)
            vector = vector.to_numpy()


    rm_indexes = np.isnan(vector[:, -1].astype(np.float))
    vector = np.delete(vector, rm_indexes, 0)
    v = torch.from_numpy(np.array(vector, dtype=np.float64)).view(-1, 1, 1041).float().to(device)
    mean = v.mean(dim=0, keepdim=True)
    std = v.std(dim=0, keepdim=True)
    normalized_v = (v - mean)/std
    normalized_v[torch.isnan(normalized_v)] = 0

    nlp_out = step4(normalized_v)
    
    processed_out = []
    i_l = []
    for idk in image_output:
        vector = idk[1]
        i_l.append(idk)
        for word in idk[0].split(' '):
            processed_out.append([word, vector])

    processed_out = processed_out


    words = words[1:]
    #nlp_out = np.reshape(nlp_out.numpy(), (int(nlp_out.shape[0]/2), -1))
    combined_out = []
    clone_nlp_out = np.copy(nlp_out.detach().numpy())
    clone_nlp_out = np.append(words.to_numpy().reshape(-1, 1), clone_nlp_out, axis=1)
    for word in processed_out:
        if word[0] != '':
            match_indexes = np.where(word[0] == clone_nlp_out[:, 0])
            if len(match_indexes[0]) == 0:
                continue
        
            match_words = clone_nlp_out[match_indexes[0][0]]
            if match_words.shape[0] > 0:
                word_string = match_words[0]
                word_vector = match_words[1:]
            combined_out.append([word_string, word_vector, word[1].numpy()])
            clone_nlp_out = np.delete(clone_nlp_out, match_indexes[0][0], axis=0)


    words = np.array(combined_out)[:, 0]
    vectors = np.array(combined_out)[:, 1:]
    #print(vectors)
    n_vectors = []
    for w in vectors:    
        n_vectors.append(np.concatenate((w[0], w[1])))

    v = torch.from_numpy(np.array(n_vectors, dtype=np.float64)).view(-1, 1, 20).float().to("cpu")
    extracted = step5(v, i_l)

    filelist = glob.glob('./tokens/*')
    for f in filelist:
        os.remove(f)
    filelist = glob.glob('./features/*')
    for f in filelist:
        os.remove(f)

    filelist = glob.glob('./feature_vectors/*')
    for f in filelist:
        os.remove(f)
    
    filelist = glob.glob('./word_lists/*')
    for f in filelist:
        os.remove(f)

    return extracted
