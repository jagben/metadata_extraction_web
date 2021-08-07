# Multimodal Approach for Metadata Extraction from German Scientific Publications Web Demo
This repository contains a simple web application for testing [model](https://github.com/azeddinebouabdallah/research-lab-ml) as a part of research lab.
## Installation
1. Clone this repository into designed directory
 ```
 git clone git@github.com:jagben/metadata_extraction_web.git
 ```
2. Change directory into the folder and install required library by using pip
```
cd metadata_extraction_web
pip install -r requirements.txt
```
3. Download [ELMO model](http://vectors.nlpl.eu/repository/11/142.zip) and extract it into "feature_extraction/context_feature_extraction/de.model"
4. Create 3 empty folder
```
mkdir features
mkdir word_lists
mkdir feature_vectors
```
## Running the demo
1. First export flask app environment variable
```
export FLASK_APP=app
```
2. Run Flask
```
flask run
```
3. without modification to the flask, this app will be host on localhost port 5000 http://127.0.0.1:5000/
