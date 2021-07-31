from .paper import *

class ModPaper(Paper):
    """
    Similar to Paper class but get_text_from_bbox do not need class
    """

    def __init__(self, filename, metadata_dict, metadata_page = 0):
        super().__init__(filename, metadata_dict, metadata_page = 0)

    def get_text_from_bbox(self, x_upper_left, y_upper_left, x_lower_right, y_lower_right, conversion=False, margin=0, use_fitz=True):
        """
        :param x_upper_left: X coordinate of the bounding box's upper left corner
        :param y_upper_left: Y coordinate of the bounding box's upper left corner
        :param x_lower_right: X coordinate of the bounding box's lower right corner
        :param y_lower_right: Y coordinate of the bounding box's lower right corner
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
                    text = self.get_text_from_bbox(x_upper_left, y_upper_left, x_lower_right, y_lower_right, conversion=False, margin=margin, use_fitz=True)
                text = text.replace("\n", "")
                text = ' '.join(text.split())

        return text
