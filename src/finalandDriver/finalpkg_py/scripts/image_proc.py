import cv2
import numpy as np
import argparse
import glob
import time

class ImageProc:
    def __init__(self, image_path):
        # self.image_path = image_path

        self.image = cv2.imread(image_path)
        # Ensure the image is in portrait mode
        if self.image.shape[1] > self.image.shape[0]:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)

    # def get_lines(self):
    #     # # Load your image
    #     # image = cv2.imread(self.image_path)

    #     # # Convert the image to grayscale
    #     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # # Apply a binary threshold to the image
    #     # _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    #     # # Find contours
    #     # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # # Draw the contours on the original image
    #     # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    #     # # Display the image with contours
    #     # cv2.imshow('Contours', image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()


    #     # Load your image
    #     image = cv2.imread(self.image_path)

    #     # Convert the image to grayscale
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     gray_image = cv2.GaussianBlur(gray_image, (5, 1), 0)

    #     # Apply Canny edge detection
    #     edges = cv2.Canny(gray_image, 255/3, 255)

    #     # Find contours
    #     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     # Draw the contours on the original image
    #     cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    #     # Create a mask where edges will be drawn
    #     edge_image = gray_image.copy()

    #     # Draw bold edges by iterating over the edge pixels
    #     edge_image[edges != 0] = 0  # Set edge pixels to black

    #     # Create a thicker edge by dilation
    #     kernel = np.ones((3, 3), np.uint8)
    #     bold_edges = cv2.dilate(edges, kernel, iterations=1)

    #     # Apply the bold edges to the grayscale image
    #     gray_image[bold_edges != 0] = 0

    #     # Display the image with bold edges
        
    #     cv2.imshow('Bold Edges on Grayscale', gray_image)

    #     # Display the image with contours

    #     cv2.imshow('Contours', image)

    #     # Convert contours to list of tuples
    #     contours_list = []
    #     for contour in contours:
    #         contour_points = [(point[0][0], point[0][1]) for point in contour]
    #         contours_list.append(contour_points)
    #         for i in range(len(contour_points)):
    #             if i != 0:
    #                 cv2.line(image, contour_points[i-1], contour_points[i], (0, 255, 0), 2)
    #             x, y = contour_points[i]
    #             # print(f"Keypoint: ({x}, {y})")

    #     print(contours_list)

    #     time.sleep(2)
    #     # cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     return contours_list
    
    def get_shape(self):
        return self.image.shape[:2]

    def scale_to_meters(self, contours_list):
        # image = cv2.imread(self.image_path)
        image = self.image
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
        # image = cv2.imread(self.image_path)
        image = self.image
        image_size = image.shape[:2]  # (height, width)

        # Convert the contours to meters
        contours_in_meters = self.scale_to_meters(contours_list, image_size)

        # Print contours in meters
        for contour in contours_in_meters:
            print(contour)

        return contours_in_meters

    def get_contours(self):
        # image = cv2.imread(self.image_path)
        image = self.image
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 255/3, 255)

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

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return contours

    def approximate_splines(self, contours, resolution=0.01):
        lines = []
        for contour in contours:
            # Approximate the contour with a spline
            epsilon = resolution * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for i in range(len(approx) - 1):
                start = tuple(approx[i][0])
                end = tuple(approx[i + 1][0])
                lines.append((start, end))
        return lines


    def otsu_lines(self):
        # # ADDED 12/4
        # # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
        # # Use a bimodal image as an input.
        # # Optimal threshold value is determined automatically.
        # # otsu_threshold, image_result = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # print("Obtained threshold: ", otsu_threshold)
        # # print()
        # # edges = cv2.Canny(image_result, otsu_threshold/3, otsu_threshold)

        # def auto_canny(image, sigma = 0.33):

        #     v = np.median(image)

        #     lower = int(max(0, (1.0 - sigma) * v))
        #     upper = int(min(255, (1.0 + sigma) * v))
        #     edged = cv2.Canny(image,lower,upper)

        #     return edged
        
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--images", required=True,
        #                 help="path to input dataset of images")
        # args = vars(ap.parse_args())

        # for imagePath in glob.glob(args["images"] + "/*.jpg"):
        #     image = cv2.imread(imagePath)
        #     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     blurred = cv2.GaussianBlur(gray_image, (3,3),0)

        #     wide = cv2.Canny(blurred, 10, 200)
        #     tight = cv2.Canny(blurred, 225, 250)
        #     auto = auto_canny(blurred)

        #     cv2.imshow("original", image)
        #     cv2.imshow("edges", np.hstack([wide, tight, auto]))
        #     cv2.waitKey(0)

        # auto_canny()


        # # END ADDED 12/4
        pass

    def get_image_size(self):
        # image = cv2.imread(self.image_path)
        return self.image.shape[:2]
