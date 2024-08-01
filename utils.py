import fitz
import cv2
import copy
import numpy as np
import easyocr
import json
import os
from datetime import datetime
from rowUtilsNew import check_header_rows_2_and_3, findTextRows, findMatchingRowPatterns



def log_message(filename, message):
    with open(filename, 'a') as file:
        file.write(message + '\n')


def get_unique_filename(directory, base_filename, extension):
    """
    Generates a unique file name by appending the current date and time,
    and if necessary, a number to avoid duplicates.
    """
    # Get the current date and time
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_filename}{extension}"
    full_path = os.path.join(directory, filename)
    
    # Check if the file already exists and append a number if it does
    counter = 1
    while os.path.exists(full_path):
        filename = f"{base_filename}({counter}){extension}"
        full_path = os.path.join(directory, filename)
        counter += 1
    
    return full_path

def openJpgImage(jpg_path):
    image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    return image

def remove_extension(file_name):
    root, ext = os.path.splitext(file_name)
    return root

# converts an OCR result into json format
# saves the json file
# and returns the path to the json file
def convert_OCR_result_to_json(result, fileName):
    output_data = []
    for detection in result:
        bounding_box, text, confidence = detection


        # Reformat the bounding box information
        bounding_box = [{'x': float(x), 'y': float(y)} for x, y in bounding_box]

        output_data.append({
            'text': text,
            'bounding_box': bounding_box,
            'confidence': confidence
        })

    
    fileName = os.path.basename(remove_extension(fileName))
    output_file_path = "OCR_Data/" + fileName + ".json"
    with open(output_file_path, 'w') as json_file:
        json.dump(output_data, json_file)

    return output_file_path


# runs easy OCR on the provided image and returns the results
def run_ocr(jpg_img_path):
    OUTPUT_IMAGE_PATH = "Temp/temp.jpg"
    pdf_document = fitz.open(jpg_img_path)
    pdf_page = pdf_document.load_page(0)
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # Convert to image with 300 DPI
    pix.save(OUTPUT_IMAGE_PATH)
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(OUTPUT_IMAGE_PATH)
    return result


# convert image to jpg 
# save the jpg
# and return path of that jpg file along with width and height values of the jpg
def pdf_to_jpg_path(pdf_path):

    SCALING_FACTOR = 300 / 72
    OUTPUT_IMAGE_PATH = "Temp/temp.jpg"
    os.makedirs("Temp", exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    pdf_page = pdf_document.load_page(0)  # Load the first page (index 0) 
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR))  # Convert to image with 300 DPI
    pix.save(OUTPUT_IMAGE_PATH)

    height = pix.height * SCALING_FACTOR
    width = pix.width  * SCALING_FACTOR
    return OUTPUT_IMAGE_PATH, width, height

# draws white rectangles over all text (as detected by the OCR read)
# in order to "erase" that text
def removeText(textImagePath, OCRData):
    textImage = cv2.imread(textImagePath, cv2.IMREAD_GRAYSCALE)
    hei, wid = textImage.shape
    blank_image3 = np.ones((hei, wid, 3), dtype=np.uint8) * 255
    for textChunk in OCRData: 
        top_left_coord = (int(textChunk["bounding_box"][0]["x"]),int(textChunk["bounding_box"][0]["y"]))
        bot_right_coord = (int(textChunk["bounding_box"][2]["x"]),int(textChunk["bounding_box"][2]["y"]))
        # "erase" all text by drawing white rectangle over it
        textless_img = cv2.rectangle(textImage, top_left_coord, bot_right_coord, (255,255,255), thickness=cv2.FILLED)
    return textless_img


# draws a white rectangle over any portion of the image
# that's above the class data
def removeTop(textImage, OCRData):
    CourseStrings = ["course", "crs", "coun5", "coursc", "courec", "cour60"]
    hei, wid = textImage.shape

    stringDetected = False
    for textChunk in OCRData:
        for course in CourseStrings:
            if(course in textChunk["text"].lower()):
                if stringDetected == False:
                    stringDetected = True
                    rowVal = textChunk["bounding_box"][0]["y"] # pixel location for bottom of "couse id" line
                    # "erase" everything above classes table
                    topless_img = cv2.rectangle(textImage, (int(0),int(0)), (int(wid), int(rowVal)), (255,255,255), cv2.FILLED)

    if stringDetected == False:
        return textImage, 0
    else:            
        return topless_img, rowVal
    


# Find instances of "state id" appearing in the data
#   *Still need to do something about transcripts 6 and 9
def removeStateID(image, OCRData, CourseHeaderRow):
    stateIDStrings = ["state id"]
    detectedStateIdPositions = []
    hei, wid = image.shape

    stringDetected = False
    for textChunk in OCRData:
        for spelling in stateIDStrings:
            if(spelling in textChunk["text"].lower()):
                if stringDetected == False:
                    stringDetected = True
                    rowVal = textChunk["bounding_box"][0]["y"] # pixel location for bottom of "couse id" line
                    detectedStateIdPositions.append(rowVal)


    numStateId = len(detectedStateIdPositions)
    img_state_id_removed = image # initialize with original image
    if numStateId == 0:
        img_state_id_removed = image
    elif  numStateId == 1:
        y_val = detectedStateIdPositions[0]
        if y_val > CourseHeaderRow:
            # "erase" everything "state id" through bottom of image
            img_state_id_removed = cv2.rectangle(image, (int(0),int(y_val)), (int(wid), int(hei)), (255,255,255), cv2.FILLED)
    else:
        for state_id_position in detectedStateIdPositions:
            if state_id_position > CourseHeaderRow:
                # "erase" everything "state id" through bottom of image
                img_state_id_removed = cv2.rectangle(img_state_id_removed, (int(0),int(y_val)), (int(wid), int(hei)), (255,255,255), cv2.FILLED)


    return img_state_id_removed    
    


# draws a white rectangle a short (but arbitrary) distance below
# the top of the student course columns
def remove_img_bottom(img, topOfClasses, imgHeight, imgWidth):
    # pixel location for bottom of "couse id" line
    fifthOfImage = imgHeight / 5
    fifthBelowClasses = topOfClasses + fifthOfImage
    columnsOnlyImage = cv2.rectangle(img, (int(0),int(fifthBelowClasses)), (int(imgWidth),int(imgHeight)), (255,255,255), thickness=cv2.FILLED)
    return columnsOnlyImage



# creates a projection profile from a white pixel projection profile
# by subtracting the current white pixel value from the
# maximum possible white pixel value
# leaving us with only black pixels
# (because image is binary)
def createBlackPixelProjectionProfile(proj_profile):


    # find max # of white pixel density in profile
    most = 0
    for colDensity in proj_profile:
        if colDensity > most:
            most = colDensity
   
    blackPixels_proj_profile = []
    for pixel_cnt in proj_profile:
         # get max white pixel minus current white pixel to find black pixel
        blackPixels_proj_profile.append(most - pixel_cnt)
    
    return blackPixels_proj_profile



def drawColumnEdges(columns, image, height, width, coursesHeaderRow, OCR_Data):


    edge1 = columns[0]
    edge2 = columns[1]
    edge3 = columns[2]
    edge4 = columns[3]

    col1Rows, col2Rows, col3Rows = findTextRows(OCR_Data, columns, coursesHeaderRow)

    row1Bot = findMatchingRowPatterns(col1Rows, coursesHeaderRow, edge2)

    row2Bot = findMatchingRowPatterns(col2Rows, coursesHeaderRow, edge3)

    row3Bot = findMatchingRowPatterns(col3Rows, coursesHeaderRow, edge4)
    

    imageCopy = copy.deepcopy(image)

    # draw vertical column edges
    cv2.rectangle(imageCopy, (int(edge1-4), int(height)),
                (int(edge1+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    cv2.rectangle(imageCopy, (int(edge2-4), int(height)),
                    (int(edge2+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    cv2.rectangle(imageCopy, (int(edge3-4), int(height)),
                    (int(edge3+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    cv2.rectangle(imageCopy, (int(edge4-4), int(height)),
                    (int(edge4+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    
    # draw top boundary line
    cv2.rectangle(imageCopy, (0, int(coursesHeaderRow - 4)),
                (int(width), int(coursesHeaderRow + 4)), (0,0,255), thickness=cv2.FILLED)
    

    # draw column bottoms:
    col2matches, col3matches = check_header_rows_2_and_3(coursesHeaderRow, columns, OCR_Data)
    # draw row 1 bottom
    cv2.rectangle(imageCopy, (int(edge1), int(row1Bot+4)),
                    (int(edge2), int(row1Bot-4)), (0,0,255), thickness=cv2.FILLED)
    # draw row 2 bottom if it exists
    if col2matches:
        cv2.rectangle(imageCopy, (int(edge2), int(row2Bot+4)),
                        (int(edge3), int(row2Bot-4)), (0,0,255), thickness=cv2.FILLED)
    # draw row 3 bottom if it exists
    if col3matches:
        cv2.rectangle(imageCopy, (int(edge3), int(row3Bot+4)),
                        (int(edge4), int(row3Bot-4)), (0,0,255), thickness=cv2.FILLED)


    return imageCopy
