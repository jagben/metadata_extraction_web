import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
from conllu import TokenList
from collections import OrderedDict
import string

from .extraction_methods.extraction_methods import *


class FeatureExtractor:

    def __init__(self, dataset_folder: str):
        self.dataset_folder = dataset_folder

    def get_feature_vector(self, document_location: str, document_number: int, file_name: str):

        # This part of code just for console progress display
        total_blocks = 40
        percentage = (document_number * 100) / 301
        num_blocks = int((percentage * total_blocks) / 100)

        print('Feature extraction - Document : {0}| Percentage: {1}{2}> : {3}%'.format(
            document_number, "#"*num_blocks, "-"*(total_blocks-num_blocks), int(percentage)), end='\r')
        words_list = []

        file = ET.parse(document_location)
        root = file.getroot()

        for element in root:
            # Feature vector for the doc
            document_vector = []
            
            if (element.tag == 'Page'):
                for zone in element:
                    if zone.tag == 'Zone':

                        # to calculate the vertical space
                        previous_bottom = 0

                        for line in zone:
                            if line.tag == 'Line':

                                # to calculate horizontal space from previous word. we only use the right position.
                                previous_right = 0

                                vertical_space = 0  # each line has different vertical space

                                for e in line:
                                    # Get the number of line.
                                    if e.tag == 'LineID':
                                        line_num = int(e.attrib['Value'])
                                    if e.tag == 'LineCorners':
                                        # used to calculate the vertical space
                                        # all words in same line will have same vertical space
                                        # y of bottom left
                                        bottom = float(e[0].attrib['y'])
                                        # y of bottom right
                                        top = float(e[3].attrib['y'])

                                        # vertical space = current line top - previous line bottom
                                        vertical_space = get_vertical_space(
                                            previous_bottom, top)

                                        previous_bottom = bottom

                                    for word in e.iter('Word'):
                                        word_vector = []  # new word vector

                                        actual_word = ''  # new word

                                        # Initializing features
                                        cap_letters = 0
                                        starts_cap = 0
                                        line_num = 0
                                        len_word = 0
                                        count_num = 0
                                        count_slash = 0  # /
                                        count_com = 0  # :
                                        is_alt = 0  # @
                                        is_email = 0
                                        is_link = 0
                                        is_year = 0
                                        is_date = 0
                                        horizantal_space = 0
                                        is_italic = 0
                                        is_bold = 0

                                        # Store positions
                                        # We need only one point for each side,
                                        # left and right for horizantal therefore only x.
                                        left = 0
                                        right = 0

                                        actual_word = ''
                                        font_size = 0

                                        for we in word:  # we: Word element
                                            if we.tag == 'WordCorners':  # get position
                                                # get the top and bottom position then substract
                                                # ceram position orders are: bottom left, bottom right, top right, top left
                                                # x of top left
                                                left = float(we[3].attrib['x'])
                                                # x of top right
                                                right = float(
                                                    we[2].attrib['x'])

                                                # horizontal space = right of previous word - left of current word
                                                horizantal_space = get_horizontal_space(
                                                    previous_right, left)

                                                # font size = bottom position - top position
                                                font_size = get_word_size(
                                                    float(we[0].attrib['y']),  float(we[3].attrib['y']))

                                                previous_right = right

                                            if we.tag == 'Character':
                                                actual_word += we[4].attrib['Value']
                                                font_type = we[3].attrib['Type']

                                                # italic or bold can be found in the font type
                                                is_italic = isItalic(font_type)
                                                is_bold = isBold(font_type)

                                        # Clean the word
                                        #actual_word = actual_word.translate(str.maketrans('', '', string.punctuation))

                                        words_list.append(actual_word)
                                        cap_letters = get_count_cap_letters(
                                            actual_word)
                                        starts_cap = starts_cap_letter(
                                            actual_word)
                                        len_word = get_word_length(actual_word)
                                        count_num = get_count_digits(
                                            actual_word)
                                        count_slash = get_count_slash(
                                            actual_word)
                                        count_com = get_count_com(actual_word)
                                        is_alt = contains_alt(actual_word)
                                        is_email = isEmail(actual_word)
                                        is_link = isLink(actual_word)
                                        is_year = isYear(actual_word)
                                        is_date = isDate(actual_word)
                                        word_vector = [
                                            cap_letters, starts_cap, line_num, len_word,
                                            count_num, count_slash, count_com,
                                            is_alt, is_email, is_link, is_year,
                                            is_date, font_size, horizantal_space, vertical_space,
                                            is_italic, is_bold
                                        ]

                                        document_vector.append(word_vector)

                # Save feature vectors as pickle files
                self.save_feature_vector(document_vector,
                                         'feature_vectors/{0}.pickle'.format(file_name.split('.')[0]))

                # Create dataframe and save csv file
                df = pd.DataFrame(document_vector, columns=['Cap letters', 'Starts cap', 'Line number', 'Len Word', 'Count num', 'Count slash', 'Count com',
                                                            'Is Alt', 'Is email', 'Is link', 'Is Year', 'Is date', 'Font size', 'Horizontal space', 'Vertical space', 'Is Italic', 'Is Bold'])
                df.insert(0, 'Word', words_list, True)
                df.to_csv('./feature_vectors/{0}vectors.csv'.format(file_name.split('.')[0]))

                self.create_dataframe(document_vector,
                                      words_list,
                                      ['Cap letters', 'Starts cap', 'Line number', 'Len Word', 'Count num', 'Count slash', 'Count com', 'Is Alt', 'Is email',
                                          'Is link', 'Is Year', 'Is date', 'Font size', 'Horizontal space', 'Vertical space', 'Is Italic', 'Is Bold'],
                                      './feature_vectors/{0}vectors.csv'.format(file_name.split('.')[0])
                                      )

                # Create CoNLL-U format file
                collnu_words_list = []
                for word in words_list:
                    collnu_words_list.append([word])
                
                collnu_list = list()
                for i, token_data in enumerate(collnu_words_list):
                    token_id = i + 1
                    token = token_data[0]
                    lemma = token_data[0]
                    compiled_token = ({'id': token_id, 'form': token, 'lemma': lemma, 'upostag': None,
                                                'xpostag': None, 'feats': None, 'head': None, 'deprel': None, 'deps': None, 'misc': None})
                    collnu_list.append(compiled_token)

                # Save words lists as pickle files
                self.save_feature_vector(collnu_words_list,
                                         'word_lists/{0}.pickle'.format(file_name.split('.')[0]))
                
                collnu_list = TokenList(collnu_list).serialize()
                with open('./tokens/tokens{0}.collnu'.format(document_number), 'w') as file:
                    file.write(collnu_list)
                    file.close()
                

    def save_feature_vector(self, vector: list, save_location: str):
        ## Create directory to avoid error
        os.makedirs('feature_vectors', exist_ok=True)
        os.makedirs('word_lists', exist_ok=True)
        os.makedirs('tokens', exist_ok=True)

        with open(save_location, 'wb') as handle:
            pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def create_dataframe(self, vectors: list, words: list, columns: list, location: str):
        df = pd.DataFrame(vectors, columns=columns)
        df.insert(0, 'Word', words, True)
        self.save_dataframe_csv(df, location)

    def save_dataframe_csv(self, df: pd.DataFrame, location: str):
        df.to_csv(location)

    def get_dataset_features(self):
        document_number = 0
        for file in os.scandir(self.dataset_folder):
            if file.path.endswith('.cermstr'):
                self.get_feature_vector(file.path, document_number, file.name)
                document_number += 1


if __name__ == '__main__':
    featureExtractor = FeatureExtractor('../../ssoar_dataset')

    featureExtractor.get_dataset_features()
