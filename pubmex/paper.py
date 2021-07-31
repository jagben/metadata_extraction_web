from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

import re

from dateparser.search import search_dates

import fitz

import base64

from io import BytesIO

import json

from itertools import groupby, chain, combinations
import math
from collections import Counter

from string import punctuation, digits, whitespace

import numpy as np

import pdfplumber

from detectron2.structures.instances import Instances

class MetadataError(Exception):
  def __init__(self, message, errors):

    # Call the base class constructor with the parameters it needs
    super().__init__(message)

    # Now for your custom code...
    self.errors = errors
        
class RectangleError(Exception):
  def __init__(self, message, errors):

    # Call the base class constructor with the parameters it needs
    super().__init__(message)

    # Now for your custom code...
    self.errors = errors

class Paper:
  """
  A wrapper for PDF documents to be used in MexPub to automatically extract metadata. 
  """
  def __init__(self, filename, metadata_dict, metadata_page=0):
    """
    :param filename: path to the pdf document
    :param metadata_dict: a dictionary with the metadata if it is already known (i.e., the PDF should be annotated based on the metadata). Keys are the classnames, values are the metadata text.
    :param metadata_page: specifies the page of the PDF from which the metadata should be extracted. Defaults to zero.
    """
    self.filename = filename
    self.metadata = metadata_dict
    self.rectangles = {}
    self.annotations = []
    self.metadata_page = metadata_page
    self.image = self.to_image()
      
  def __str__(self):
    return self.filename
  
  def to_image(self):
    img = convert_from_path(self.filename)[self.metadata_page]
    return img
  
  def resize_image(self, width=569, height=794):
    """
    :param width: indicates the width the resized image should have
    :param height: indicates the height the resized image should have
    """
    self.image = self.image.resize((width, height))
      
  
  def save_image(self,output_path=""):
    """
    :param output_path: sepcifies the path to which the image is saved
    """
    reg = re.compile("[^/]+$")
    img_name = re.search(reg, self.filename)[0][:-4] + ".jpeg"
    if len(output_path) != 0 and output_path[-1] != "/":
        output_path += "/"
    self.image.save(output_path + img_name)
      
  def get_metadata_items(self, metadata_df):
    """
    :param metadata_df: a pandas DataFrame containing the metadata for a given PDF
    """
    reg = re.compile("[^/]+$")
    instance = re.search(reg, self.filename)[0][:-4] + ".docx"
    
    current = metadata_df[metadata_df["filenames"] == instance]
    for key in self.metadata.keys():
        self.metadata[key] += str(list(current[key])[0])
      
  def bbox_from_text(self):
      # get the first page
      doc = fitz.open(self.filename)
      page = doc[self.metadata_page]
      
      if list(self.metadata.values())[0] == "":
          raise MetadataError("No metadata values specified. Run method get_metadata_items(metadata_df) before callsing this method.", "")
      i = 0
      for key, val in self.metadata.items():
          self.rectangles[key] = page.searchFor(val, hit_max=1000)
          # handle hyphenated words
          if len(self.rectangles[key]) == 0:
              text = self.metadata[key].split("-")
              for part in text:
                  results = page.searchFor(part, hit_max=1000)
                  self.rectangles[key] += results
          # if more than one Rect was found, combine them
          if len(self.rectangles[key]) > 1:
              foundRectList = [tuple(rect) for rect in self.rectangles[key]]
  
              #find min x, min y, max x, max y
              min_x = foundRectList[0][0]
              min_y = foundRectList[0][1]
              max_x = foundRectList[0][2]
              max_y = foundRectList[0][3]

              for rect in foundRectList:
                  if rect[0] < min_x:
                      min_x = rect[0]
                  if rect[1] < min_y:
                      min_y = rect[1]
                  if rect[2] > max_x:
                      max_x = rect[2]
                  if rect[3] > max_y:
                      max_y = rect[3]
          
              self.rectangles[key] = [fitz.Rect(min_x, min_y, max_x, max_y)]
          
          #check if the correct text was found
          # """
          # if len(self.rectangles[key]) == 0:#key == "abstract" and self.get_text_from_bbox(self.rectangles[key][0][0],self.rectangles[key][0][1], self.rectangles[key][0][2], self.rectangles[key][0][3]):                
          #     textBlocks = page.getTextBlocks()
          #     foundRectList = [fitz.Rect(block[:4]) for block in textBlocks if get_cosine_similarity(str.lower(block[4]), str.lower(val)) > 0.7]
          #     #find min x, min y, max x, max y
          #     min_x = foundRectList[0][0]
          #     min_y = foundRectList[0][1]
          #     max_x = foundRectList[0][2]
          #     max_y = foundRectList[0][3]

          #     for rect in foundRectList:
          #         if rect[0] < min_x:
          #             min_x = rect[0]
          #         if rect[1] < min_y:
          #             min_y = rect[1]
          #         if rect[2] > max_x:
          #             max_x = rect[2]
          #         if rect[3] > max_y:
          #             max_y = rect[3]

          #     self.rectangles[key] = [fitz.Rect(min_x, min_y, max_x, max_y)]
          # """
              
  def get_annotations(self):
    if len(self.rectangles) == 0:
        raise RectangleError("No rectangles available. Run method bbox_from_text(self) before calling this method.", "")
    
    doc = fitz.open(self.filename)
    page = doc[self.metadata_page]
    
    version = "4.4.0"
    flags = {}
    shapes = []
    reg = re.compile("[^/]+$")
    imagePath = re.search(reg, self.filename)[0][:-4] + ".jpeg"
    
    buffered = BytesIO()
    self.image.save(buffered, format="JPEG")
    imageDataBase64 = base64.b64encode(buffered.getvalue())
    imageData = str(imageDataBase64)[1:].replace("'","")
    
    imageHeight = self.image.height
    imageWidth = self.image.width
    
    for key, val in self.rectangles.items():
        annotation = {
            "label": key,
            "points": [],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        
        #convert the x and y values of the rectangles
        rect_points = val[0]
        
        
        x_conversion = float(self.image.width / page.rect.width)
        y_conversion = float(self.image.height / page.rect.height)
        width = (rect_points[2] - rect_points[0]) * x_conversion
        height = (rect_points[3] - rect_points[1]) * y_conversion
        
        rect_points_converted = [
            rect_points[0] * x_conversion,
            rect_points[1] * y_conversion,
            rect_points[2] * x_conversion,
            rect_points[3] * y_conversion
        ]
        
        annotation["points"]= [
            [rect_points_converted[0], rect_points_converted[1]],
            [rect_points_converted[0] + width, rect_points_converted[1]],
            [rect_points_converted[0] + width, rect_points_converted[3]],
            [rect_points_converted[0], rect_points_converted[3]]
        ]
        shapes.append(annotation)
    annotations = {
        "version": version,
        "flags": flags,
        "shapes": shapes,
        "imagePath": imagePath,
        "imageData": imageData,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }
    self.annotations = annotations
    return annotations
  
  def save_annotations(self, output_path=""):
    """
    :param output_path: specifies the folder to which the annotations are saved. 
    """
    if len(self.annotations) == 0:
        self.get_annotations()
    
    if len(output_path) != 0 and output_path[-1] != "/":
        output_path += "/"
        
    reg = re.compile("[^/]+$")
    outPath = re.search(reg, self.filename)[0][:-4] + ".json"
    with open(output_path + outPath, "w") as f:
        json.dump(self.annotations, f)
          
  def get_text_from_bbox(self, x_upper_left, y_upper_left, x_lower_right, y_lower_right, class_name, conversion=False, margin=0, use_fitz=True):
    """
    :param x_upper_left: X coordinate of the bounding box's upper left corner
    :param y_upper_left: Y coordinate of the bounding box's upper left corner
    :param x_lower_right: X coordinate of the bounding box's lower right corner
    :param y_lower_right: Y coordinate of the bounding box's lower right corner
    :param class_name: specifies the class name of the given bounding box (e.g., title)
    :param conversion: specifies whether the coordinates of the bounding box have to be converted to match the PDF document's size
    :param margin: specifies whether a margin should be added to the bounding box when extracting the text
    :param use_fitz: specifies whether the method uses the fitz-library to extract the text. If set to False, the method uses pdfplumber
    """
    doc = fitz.open(self.filename)
    page = doc[self.metadata_page]
    if conversion:
      x_conversion = float(page.rect.width / self.image.width)
      y_conversion = float(page.rect.height / self.image.height)
      x_upper_left = (x_upper_left * x_conversion)
      y_upper_left = (y_upper_left * y_conversion)
      x_lower_right = (x_lower_right * x_conversion)
      y_lower_right = (y_lower_right * y_conversion)
    
    text = ""

    if use_fitz:
      rect = fitz.Rect(x_upper_left-margin, y_upper_left-margin, x_lower_right+margin, y_lower_right+margin)

      self.rectangles[class_name] = rect
      
      words = page.getText("words") # list of words on the page
      words.sort(key=lambda w: (w[3], w[0])) # ascending y, then x coordinate

      # sub-select only words that are contained INSIDE the rectangle
      mywords = [w for w in words if fitz.Rect(w[:4]).intersects(rect)]
      #mywords = [w for w in words if fitz.Rect(w[:4]) in rect]
      group = groupby(mywords, key=lambda w: w[3])

      for _, gwords in group:
          text += " ".join(w[4] for w in gwords)

    else: 
      with pdfplumber.open(self.filename) as pdf:
        first_page = pdf.pages[self.metadata_page]
        text = first_page.crop((x_upper_left, y_upper_left, x_lower_right, y_lower_right)).extract_text(x_tolerance=margin, y_tolerance=margin)
        if text is None:
          text = self.get_text_from_bbox(x_upper_left, y_upper_left, x_lower_right, y_lower_right, class_name, conversion=False, margin=margin, use_fitz=True)
        text = text.replace("\n", "")
        text = ' '.join(text.split())

    return text

  def get_text_from_detectron2_outputs(self, detectron2_instances, metadataCatalog, margin=0, use_fitz=True):
    """
    :param detectron2_instances: a Tensor object, i.e., the instances output by the MexPub predictor
    :param metadataCatalog: the detectron2 metadata catalog registered to infer the class names
    :param margin: specifies the margin to be added to the bounding boxes when extracting the text
    :param use_fitz: specifies whether the method uses the fitz-library to extract the text. If set to False, the method uses pdfplumber
    """
    if len(detectron2_instances) == 0:
      print("No instances detected.")
      return
    scores = {}
    metadata = {}
    for i in range(len(detectron2_instances)):
      instance = detectron2_instances[i]
      fields = instance._fields
      class_name = metadataCatalog.thing_classes[fields["pred_classes"].item()]
      bbox = fields["pred_boxes"].tensor.cpu().numpy()[0]
      if class_name not in self.rectangles.keys():
        self.rectangles[class_name] = [fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])]
      else:
        self.rectangles[class_name].append(fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3]))
    self.post_process_rectangles()
    for key, val in self.rectangles.items():
      # sort the rectangles from top to bottom
      val = sorted(val, key=lambda rect: rect[1])
      if type(val) != list:
        val = [val]
      for v in val:
        text = self.get_text_from_bbox(v[0], v[1], v[2], v[3], class_name, conversion=True, margin=margin, use_fitz=use_fitz)
        if not key in metadata.keys():
          metadata[key] = text
        else:
              metadata[key] += "\n" + text
    """      text = self.get_text_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], class_name, conversion=True,margin=margin, use_fitz=use_fitz)
          if not class_name in scores.keys(): #or scores[class_name] < fields["scores"].item():
            # get the text  
            metadata[class_name] = text
            scores[class_name] = fields["scores"].item()
          else:
            metadata[class_name] += "/--/" + text
    """      
    self.metadata = metadata
    return metadata

  def post_process_rectangles(self):
    """
      merge rectangles if they have the same class label and are close to each other
    """
    for key, val in self.rectangles.items():
      if len(val) > 1:
        new_rect = None
        merged = False
        # compare each rectangle with each other rectangle
        for a, b in combinations(val, 2):
          new_rect = None
          relative_distance_y = abs(b[3]/self.image.height - a[1]/self.image.height)
          relative_distance_x1 = abs(b[0]/self.image.width - a[0]/self.image.width)

          #check if the rectangles are close to each other and there is no other rectangle between them
          rectangles = self.rectangles.copy()
          # order the rectangles by the y coordinate of the bounding box's top left corner
          rectangles = sorted([rect for rects in rectangles.values() for rect in rects], key=lambda rect: rect[1])
          if relative_distance_y < 0.1 and relative_distance_x1 < 0.2 and abs(rectangles.index(a) - rectangles.index(b)) == 1:
            # merge the two rectangles
            new_rect = fitz.Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            merged = True
            #new_rects.append(new_rect)
          #else:
            #new_rects.append(a)
            # new_rects.append(b)
          
          if new_rect != None:
            self.rectangles[key].append(new_rect)

            self.rectangles[key].remove(a)
            self.rectangles[key].remove(b)

            self.post_process_rectangles()
            break

  def post_process_metadata(self):
    #method to apply NLP to the metadata field
    for class_name in self.metadata.keys():

      if class_name == "abstract":
        #remove superscripts at the end of the abstract
        superscript_end = r'[0-9]+$'
        end_result = re.findall(superscript_end, str(self.metadata[class_name]))
        if len(end_result) != 0 and len(end_result[0]) == 1:
          self.metadata["abstract"] = re.sub(superscript_end, "", str(self.metadata[class_name]))

      elif class_name == "title":
        #remove superscripts at the beginning and the end of the title
        superscript_start = r'^[0-9]+'
        superscript_end = r'[0-9]+$'
        
        start_result = re.findall(superscript_start, str(self.metadata[class_name]))
        end_result = re.findall(superscript_end, str(self.metadata[class_name]))
        if len(start_result) != 0 and len(start_result[0]) == 1:
          self.metadata["title"] = re.sub(superscript_start, "", str(self.metadata[class_name]))
        if len(end_result) != 0 and len(end_result[0]) == 1:
          self.metadata["title"] = re.sub(superscript_end, "", str(self.metadata[class_name]))

      elif class_name == "date":
        #strip punctuation and whitespace
        self.metadata["date"] = str(self.metadata[class_name]).strip(punctuation + whitespace)
        try:
          self.metadata["date"] = search_dates(self.metadata["date"])[0][0]
        except:
          pass
        
      elif class_name == "email":
        regex = r'([\w0-9._-]+@[\w0-9._-]+\.[\w0-9_-]+)'
        self.metadata["email"] = ";".join(re.findall(regex, str(self.metadata[class_name]), re.M|re.I))

      elif class_name == "doi":
        # strip punctuation and whitespace
        self.metadata["doi"] = str(self.metadata[class_name]).strip(punctuation + whitespace)

      elif class_name == "author":
        # strip punctuation, digits, and whitespace
        self.metadata["author"] = str(self.metadata[class_name]).lstrip(punctuation + digits + whitespace)
        self.metadata["author"] = str(self.metadata[class_name]).rstrip(whitespace + digits)

      elif class_name == "affiliation":
        # strip punctuation and whitespace
        self.metadata["affiliation"] = str(self.metadata[class_name]).lstrip(whitespace + punctuation)

      elif class_name == "journal":
        # strip punctuation and whitespace
        self.metadata["journal"] = str(self.metadata[class_name]).lstrip(whitespace + punctuation)

      elif class_name == "address":
        # strip punctuation and whitespace
        self.metadata["address"] = str(self.metadata[class_name]).lstrip(whitespace + punctuation)

    
# helper_functions to compute cosine similarity between two strings
def get_cosine(vec1, vec2):
  """
  :param vec1: the first vector
  :param vec2: the second vector
  """
  intersection = set(vec1.keys()) & set(vec2.keys())
  numerator = sum([vec1[x] * vec2[x] for x in intersection])

  sum1 = sum([vec1[x]**2 for x in vec1.keys()])
  sum2 = sum([vec2[x]**2 for x in vec2.keys()])
  denominator = math.sqrt(sum1) * math.sqrt(sum2)

  if not denominator:
      return 0.0
  else:
      return float(numerator) / denominator


def text_to_vector(text):
  """
  :param text: a string that should be converted to a vector
  """
  word = re.compile(r'\w+')
  words = word.findall(text)
  return Counter(words)


def get_cosine_similarity(content_a, content_b):
  """
  :param content_a: string
  :param content_b: string
  """
  text1 = content_a
  text2 = content_b

  vector1 = text_to_vector(text1)
  vector2 = text_to_vector(text2)

  cosine_result = get_cosine(vector1, vector2)
  return cosine_result
  