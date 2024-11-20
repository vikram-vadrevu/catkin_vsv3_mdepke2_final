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

        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
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
        cv2.imshow('Bold Edges on Grayscale', gray_image)
        # Display the image with contours
        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


