import os
import cv2
import pytesseract
import re
import numpy as np
import matplotlib.pyplot as plt


# Function to write sheet information to a text file
def write_sheet_info(filename, corners, sheet_id, output_file):
    with open(output_file, 'a') as file:
        line = f"{filename},{corners[0][0]},{corners[0][1]},{corners[1][0]},{corners[1][1]},"
        line += f"{corners[2][0]},{corners[2][1]},{corners[3][0]},{corners[3][1]},{sheet_id}\n"
        file.write(line)


def visualize_lines(binary_mask, lines):
    # Create an empty image to draw lines on
    lines_image = np.zeros_like(binary_mask)

    # Draw lines on the empty image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)

    # Visualize the lines on the binary mask
    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    output_image[lines_image > 0] = [0, 255, 0]  # Highlight lines in green

    # Display the result
    plt.imshow(output_image)
    plt.title('Detected Lines')
    plt.show()


def visualize_horizontal_vertical_lines(binary_mask, lines):
    # Create an empty image to draw lines on
    lines_image = np.zeros_like(binary_mask)

    # Draw lines on the empty image
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)

    # Visualize the lines on the binary mask
    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    output_image[lines_image > 0] = [0, 255, 0]  # Highlight lines in green

    # Display the result
    plt.imshow(output_image)
    plt.title('Detected horizontal vertical lines')
    plt.show()


def visualize_mask(original_image, binary_mask):
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Display the binary mask
    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')  # Use 'gray' colormap for binary images
    plt.title('Binary Mask')

    plt.show()


def extract_sheet_id_from_text(text):
    # Extract sheet ID based on the provided pattern
    pattern = re.compile(r'[A-Z]?-[1-9][0-9]?-[1-9][0-9]?-[A-D]?-[a-d]?')
    match = re.search(pattern, text)
    return match.group() if match else None


def extract_sheet_id(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define the region where the sheet ID is expected (you may need to adjust these coordinates)
    top_frame_region = image[0:300, 0:image.shape[1]]

    # Convert the region to grayscale
    gray = cv2.cvtColor(top_frame_region, cv2.COLOR_BGR2GRAY)

    # Use pytesseract to extract text from the cropped region
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config)

    # Extract sheet ID using the defined pattern
    sheet_id = extract_sheet_id_from_text(text)

    return sheet_id


def find_inner_corners(image_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = img.shape

    # Calculate crop dimensions
    crop_percent = 5.0 / 100.0
    crop_height = int(height * crop_percent)
    crop_width = int(width * crop_percent)

    # Crop the image
    cropped_image = img[crop_height:-crop_height, crop_width:-crop_width]

    cropped_height, cropped_width, _ = cropped_image.shape

    lower_threshold = np.array([0, 0, 0], dtype=np.uint8)
    upper_threshold = np.array([127, 127, 127], dtype=np.uint8)
    black_mask = cv2.inRange(cropped_image, lower_threshold, upper_threshold)
    binary_mask = black_mask.astype(np.uint8) * 255

    lines = cv2.HoughLinesP(binary_mask, 1, theta=np.pi / 180 / 3, threshold=int(cropped_width / 4),
                            minLineLength=(cropped_width / 8), maxLineGap=10)

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * (180 / np.pi)

        if 0 <= abs(angle) <= 5:
            horizontal_lines.append(line[0])

        elif 85 <= abs(angle) <= 95:
            vertical_lines.append(line[0])

    # Sort vertical lines based on x-coordinate to get leftmost and rightmost lines
    vertical_lines.sort(key=lambda x: (x[0], x[2]))

    leftmost_vertical_line = vertical_lines[0]
    rightmost_vertical_line = vertical_lines[-1]

    # Extract topmost horizontal line
    if horizontal_lines:
        topmost_horizontal_line = min(horizontal_lines, key=lambda x: x[1])

        outer_left_top_corner = (leftmost_vertical_line[0], topmost_horizontal_line[1])
        outer_right_top_corner = (rightmost_vertical_line[2], topmost_horizontal_line[1])

        estimated_distance = ((outer_right_top_corner[0] - outer_left_top_corner[0]) ** 2 + (
                outer_right_top_corner[1] - outer_left_top_corner[1]) ** 2) ** 0.5

        # Estimate the rough position of the bottom frame
        bottom_frame_y = estimated_distance * 1.03

        filtered_horizontal_lines = [line for line in horizontal_lines if line[3] <= bottom_frame_y]

        # Find the bottommost horizontal line
        bottom_horizontal_line = max(filtered_horizontal_lines, key=lambda line: line[3])

        map_frame_height_pixels = bottom_horizontal_line[1] - topmost_horizontal_line[1]

        inner_height_mm = 371
        frame_width_mm = 10
        pixels_in_one_mm = map_frame_height_pixels // inner_height_mm
        offset = pixels_in_one_mm * frame_width_mm

        left_top_corner = (leftmost_vertical_line[0] + offset + crop_width,
                           topmost_horizontal_line[1] + offset + crop_height)

        right_top_corner = (rightmost_vertical_line[2] - offset + crop_width,
                            topmost_horizontal_line[1] + offset + crop_height)

        left_bottom_corner = (leftmost_vertical_line[0] + offset + crop_width,
                              bottom_horizontal_line[3] - offset + crop_height)

        right_bottom_corner = (rightmost_vertical_line[2] - offset + crop_width,
                               bottom_horizontal_line[3] - offset + crop_height)


        # # Visualize the lines on the original image
        # output_image = img.copy()  # Assuming `img` is the original image
        #
        # # Apply circles on the original image
        # cv2.circle(output_image, left_top_corner, 12, (255, 0, 255), 3)
        # cv2.circle(output_image, right_top_corner, 12, (255, 0, 255), 3)
        # cv2.circle(output_image, left_bottom_corner, 12, (255, 0, 255), 3)
        # cv2.circle(output_image, right_bottom_corner, 12, (255, 0, 255), 3)
        #
        # # Display the result
        # # Set the figure size (adjust the values as needed)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(output_image)
        # plt.title('Detected Lines and Corners')
        # plt.show()

        return (left_top_corner , right_top_corner, right_bottom_corner, left_bottom_corner)

    else:
        print("No horizontal lines found.")


def process_maps():
    path = input("Enter the path where the scanned files are: ")

    files = os.listdir(path)
    # Iterate over files in the given path
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(path, filename)
            # Extract sheet ID
            sheet_id = extract_sheet_id(image_path)
            # Find corners
            corners = find_inner_corners(image_path)

            if corners:
                write_sheet_info(filename,
                                 [corners[0], corners[1], corners[2], corners[3]],
                                 sheet_id, "output.txt")
            else:
                print("No corners found.")


if __name__ == "__main__":
    process_maps()
