# import libraries
import os
import argparse
import json
from utils import pdf_to_jpg_path, removeText, remove_img_bottom, removeTop, removeStateID, \
    createBlackPixelProjectionProfile, run_ocr, convert_OCR_result_to_json, \
        openJpgImage, drawColumnEdges, get_unique_filename, remove_extension
from rowUtilsNew import findTextRows
from FinalizeColumns import check_predicted_column_values
from English_Learner_Detector import check_for_english_learner
from tilt_manager import find_angles, determine_orientation, rotate_image
import cv2
import numpy as np

def process_single_image(file_name, input_folder_path):
    print("processing image " + str(file_name) + " in " + str(input_folder_path))
    english_learner, \
    aligned_image, \
    angle_rotated, \
    course_data_edges_drawn \
        = process_image(file_name, input_folder_path)
    

    name = remove_extension(file_name)

    # Define the directory path you want to create
    output_folder_path = 'Single_Transcripts/' + str(name)

    # Create the directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)


    
    # Now you can save your file in the directory
    file_path = get_unique_filename(output_folder_path, name, '.txt')
    with open(file_path, 'w') as file:
        file.write(str(file_name) + " = " + str(english_learner) + str("\n"))

    rotated_img_file_name = str(name) + "_angle=" + str(angle_rotated) + ".jpg"
    rotated_img_path = str(output_folder_path) + "/" + str(rotated_img_file_name)
    cv2.imwrite(rotated_img_path, aligned_image)
    
    edges_drawn_file_name = str(name) + "_edges.jpg"
    edges_drawn_path = str(output_folder_path) + "/" + str(edges_drawn_file_name)
    cv2.imwrite(edges_drawn_path, course_data_edges_drawn)




# should return boolean for english learner status - ( turn into .txt file / files later... )
# return un-tilted image  with tilt degrees in name
# return column edges image
#   col top, col edges, col bottoms?
def process_image(filename, input_folder_path):

    print("processing transcript " + str(filename) + " in folder " + str(input_folder_path))

    file_path = os.path.join(input_folder_path, filename)

    # convert image to jpg 
    original_jpg_path, width, height = pdf_to_jpg_path(file_path)
    # do ocr read if necessary
    result = run_ocr(original_jpg_path)
    OCR_Data_Path = convert_OCR_result_to_json(result, filename)

    with open(OCR_Data_Path, 'r') as json_file:
        OCR_Data = json.load(json_file)

    originalImg = openJpgImage(original_jpg_path)
    textlessImg = removeText(original_jpg_path, OCR_Data)
    toplessImg, coursesHeaderRow = removeTop(textlessImg, OCR_Data)


    # REDACTED IMAGE
    cleanImage = cv2.imread(original_jpg_path, cv2.IMREAD_GRAYSCALE)
    semi_redacted_image, coursesHeaderRow = removeTop(cleanImage, OCR_Data)
    redacted_image = removeStateID(semi_redacted_image, OCR_Data, coursesHeaderRow)





    bottomlessImg = remove_img_bottom(toplessImg, coursesHeaderRow, height, width)

    projection_profile = np.sum(bottomlessImg, axis=0)

    blackPixProjProfile = createBlackPixelProjectionProfile(projection_profile)

    columns = check_predicted_column_values(blackPixProjProfile, original_jpg_path, height)

    col1Rows, col2Rows, col3Rows = findTextRows(OCR_Data, columns, coursesHeaderRow)

    rows = []
    for row in col1Rows:
        rows.append(row)
    for row in col2Rows:
        rows.append(row)
    for row in col3Rows:
        rows.append(row)

    english_learner = check_for_english_learner(rows)

    angles = find_angles(textlessImg)

    angle_rotated = determine_orientation(angles)



    aligned_image = rotate_image(redacted_image, -angle_rotated)

    course_data_edges_drawn = drawColumnEdges(columns, redacted_image, height, width, coursesHeaderRow, OCR_Data)

    # Remove temporary image file
    OUTPUT_IMAGE_PATH = "Temp/temp.jpg"
    if os.path.exists(OUTPUT_IMAGE_PATH):
        os.remove(OUTPUT_IMAGE_PATH)


    return english_learner, aligned_image, angle_rotated, course_data_edges_drawn


def process_images_in_folder(folder_path):
    print("processing all images in " + str(folder_path) + " folder")
    


    English_Learner_All = ""

     # List all files in the folder
    for filename in os.listdir(folder_path):
        try:
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Check if the file is an image
            if filename.lower().endswith(('.pdf')):
                
                print("processing image " + str(filename) + " in " + str(folder_path))
                english_learner, \
                aligned_image, \
                angle_rotated, \
                course_data_edges_drawn \
                    = process_image(filename, folder_path)
                

                name = remove_extension(filename)

                # Define the directory path you want to create
                output_folder_path = 'Processed_Transcripts/' + str(name)

                # Create the directory if it doesn't exist
                os.makedirs(output_folder_path, exist_ok=True)


                
                # Now you can save your file in the directory
                file_path = get_unique_filename(output_folder_path, name, '.txt')
                with open(file_path, 'w') as file:
                    file.write(str(filename) + " = " + str(english_learner) + str("\n"))

                rotated_img_file_name = str(name) + "_angle=" + str(angle_rotated) + ".jpg"
                rotated_img_path = str(output_folder_path) + "/" + str(rotated_img_file_name)
                cv2.imwrite(rotated_img_path, aligned_image)
                
                edges_drawn_file_name = str(name) + "_edges.jpg"
                edges_drawn_path = str(output_folder_path) + "/" + str(edges_drawn_file_name)
                cv2.imwrite(edges_drawn_path, course_data_edges_drawn)

                English_Learner_All += ( str(filename) + " = " + str(english_learner) + "\n" )
        except Exception as e:
            print(f"An error occurred: {e}")
            continue 
            
    # ----------------vvvvvvvvvvvvvvvvvv------------------#

    # Create the directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)


    
    # Now you can save your file in the directory
    output_folder_path = "Processed_Transcripts"
    eng_learner_name_all = "All_English_Learners"
    file_path = get_unique_filename(output_folder_path, eng_learner_name_all, '.txt')
    with open(file_path, 'w') as file:
        file.write(English_Learner_All)
    






def main():
    parser = argparse.ArgumentParser(description="Run OCR on images.")
    parser.add_argument('command', choices=['run'], help="The command to run.")
    parser.add_argument('target', help="The target to process: 'all' for all images in the folder or the specific image filename.")
    parser.add_argument('folder', help="The path to the folder containing images.")
    
    args = parser.parse_args()
    
    if args.command == 'run':
        if args.target == 'all':
            process_images_in_folder(args.folder)
        else:
            image_path = os.path.join(args.folder, args.target)
            if os.path.isfile(image_path):
                process_single_image(args.target, args.folder)
            else:
                print(f"File {args.target} not found in folder {args.folder}")

if __name__ == "__main__":
    main()




