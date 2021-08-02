import subprocess
import os

class CermineTextExtractor:

    def __init__(self):
        self.current_location = os.path.dirname(__file__)
        self.folder_location = ""
        self.jar_location = "cermine-impl-1.13-jar-with-dependencies.jar"

    # Extracts texts of all pdfs in folder and adds the corresponding extraction files to the source folder  
    def cermine_extract(self, folder_location, zonesOption, truevizOption):

        print("self.current_location = ",self.current_location)
        if self.current_location.isspace():
            jar_path = self.jar_location
        else:
            jar_path = self.current_location+"/"+self.jar_location

        command = 'java -cp '+ jar_path + ' pl.edu.icm.cermine.ContentExtractor'
        command += ' -path ' + folder_location
        command += ' -outputs text'

        if zonesOption:
            command += ',zones'
        if truevizOption:
            command += ',trueviz'

        print("Run command: " + command)

        subprocess.call(command, shell=True) # Use call to make it synchonize
        
        return   

if __name__ == "__main__":
    textExtractor = CermineTextExtractor()

    # Will only output something if there is a new, unprocessed pdf in the folder
    # Also it doesn't work if there are spaces in foldernames in the path
    textExtractor.cermine_extract('temp/',True, True)

    # java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path ../temp/ -outputs text
    # java -cp /cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path ../temp/ -outputs text,zones,trueviz

