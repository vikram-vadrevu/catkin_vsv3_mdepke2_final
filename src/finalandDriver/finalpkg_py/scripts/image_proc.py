import cv2
import numpy as np

class ImageProc:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_lines(self):
        # # Load your image
        # image = cv2.imread(self.image_path)

        # # Convert the image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Apply a binary threshold to the image
        # _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

        # # Find contours
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw the contours on the original image
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # # Display the image with contours
        # cv2.imshow('Contours', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Load your image
        image = cv2.imread(self.image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_image = cv2.GaussianBlur(gray_image, (5, 1), 0)
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 75, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the original image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Create a mask where edges will be drawn
        edge_image = gray_image.copy()

        # Draw bold edges by iterating over the edge pixels
        edge_image[edges != 0] = 0  # Set edge pixels to black

        # Create a thicker edge by dilation
        kernel = np.ones((3, 3), np.uint8)
        bold_edges = cv2.dilate(edges, kernel, iterations=1)

        # Apply the bold edges to the grayscale image
        gray_image[bold_edges != 0] = 0

        # Display the image with bold edges

        # cv2.imshow('Bold Edges on Grayscale', gray_image)

        # Display the image with contours

        # cv2.imshow('Contours', image)


        # Convert contours to list of tuples
        contours_list = []
        for contour in contours:
            contour_points = [(point[0][0], point[0][1]) for point in contour]
            contours_list.append(contour_points)
            for i in range(len(contour_points)):
                if i != 0:
                    cv2.line(image, contour_points[i-1], contour_points[i], (0, 255, 0), 2)
                x, y = contour_points[i]
                # print(f"Keypoint: ({x}, {y})")

        # Print all contours as lists of tuples
        # for contour in contours_list:
        #     print(contour)

        print(contours_list)


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return contours_list
    

    def scale_to_meters(self, contours_list):
        image = cv2.imread(self.image_path)
        image_size = image.shape[:2]
        # Define the dimensions of the paper in meters
        paper_width = 0.2159  # 8.5 inches in meters
        paper_height = 0.2794 # 11 inches in meters

        # Calculate the scaling factors
        scale_x = paper_width / image_size[1]
        scale_y = paper_height / image_size[0]

        # Convert pixel coordinates to meters
        contours_in_meters = []
        for contour in contours_list:
            contour_in_meters = [(x * scale_x, y * scale_y) for x, y in contour]
            contours_in_meters.append(contour_in_meters)

        return contours_in_meters

    def get_lines_and_scale(self):
        # Get pixel-wise contours
        contours_list = self.get_lines()

        # Load the image to get its size
        image = cv2.imread(self.image_path)
        image_size = image.shape[:2]  # (height, width)

        # Convert the contours to meters
        contours_in_meters = self.scale_to_meters(contours_list, image_size)

        # Print contours in meters
        for contour in contours_in_meters:
            print(contour)

        return contours_in_meters

