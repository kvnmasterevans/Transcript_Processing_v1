import cv2
import numpy as np

# Load an image


def find_angles(image):


    edges = cv2.Canny(image, 50, 150)

    detected_angles = {}

    # Standard Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 360, 200)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            angle_deg = np.degrees(theta) % 180  # Convert radians to degrees
            angle_key = round(angle_deg, 2)  # Round to 2 decimal places for grouping
            if angle_key not in detected_angles:
                detected_angles[angle_key] = 1
            else:
                detected_angles[angle_key] += 1


    
    # for line in lines:
    #     angle = str(line[1])
    #     if line[1] not in detected_angles:    
    #         detected_angles[angle] = 1
    #     elif line[1] in detected_angles:
    #         detected_angles[angle] += 1

    return detected_angles



# greatest_angle = None
# second_greatest_angle = None
# def determine_orientation(detected_angles):
#     for angle in detected_angles:
#         # ???

def determine_orientation(detected_angles):
    sorted_angles = sorted(detected_angles.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_angles) > 1:
        greatest_angle = sorted_angles[0][0]
        second_greatest_angle = sorted_angles[1][0]
    else:
        greatest_angle = sorted_angles[0][0]
        second_greatest_angle = None


    vertical_diff_1 = 90 - greatest_angle
    vertical_diff_2 = 90 - second_greatest_angle

    smaller_angle_diff = min(vertical_diff_1, vertical_diff_2, key = abs)

    return smaller_angle_diff



def rotate_image(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Compute the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
    return rotated