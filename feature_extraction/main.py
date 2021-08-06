from PyPDF2 import PdfFileWriter, PdfFileReader
import os

from layout_feature_extraction.main import FeatureExtractor
from context_feature_extraction.main import ContextFeatureExtractor


class DataExtractor:

    def __init__(self):
        self.current_path = os.path.dirname(__file__)
        self.dataset_folder_name = self.current_path + '/../ssoar_downloads'
        self.result_folder = self.current_path + '/../ssoar_dataset/'
        self.num_files = 0

    def extract_data(self):
        for file in os.scandir(self.dataset_folder_name):
            if (file.path.endswith('.pdf')) and file.is_file():

                pdf = PdfFileReader(file.path)
                pdfWriter = PdfFileWriter()
                pdfWriter.addPage(pdf.getPage(1))
                new_path = self.result_folder + file.name
                print(new_path)
                f = open(self.result_folder +  file.name, 'wb')
                pdfWriter.write(f)
                f.close()
                self.num_files += 1

if __name__ == '__main__':

    # Extract layout features
    layoutFeatureExtractor = FeatureExtractor('testpdfs')
    layoutFeatureExtractor.get_dataset_features()

    contextFeatureExtractor = ContextFeatureExtractor("word_lists/", "features/")
    contextFeatureExtractor.get_features()

    
