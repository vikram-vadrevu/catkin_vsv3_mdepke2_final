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

    def get_shape(self):
        return self.image.shape[:2]


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

    def get_image_size(self):
        # image = cv2.imread(self.image_path)
        return self.image.shape[:2]
    

    def get_contours_filter_exp(self, epsilon_factor=0.01, distance_threshold=10, min_length=50):
        """Get contours from the image with reduced unnecessary line segments, merged nearby contours, and filter out short contours.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        epsilon_factor : float, optional
            Approximation accuracy. Default is 0.01.
        distance_threshold : int, optional
            Distance threshold to consider two contours close. Default is 10.
        min_length : int, optional
            Minimum length of the contours to keep. Default is 50.

        Returns
        -------
        filtered_contours : list of np.ndarray
            The contours detected in the image, with close contours merged and short contours filtered out.
        """
        image = self.image
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 100)

        # Use Hough Line Transform to refine edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_length, maxLineGap=distance_threshold)

        # Create a blank mask to draw the lines
        line_image = np.zeros_like(image)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Combine the lines with the original edges
        combined_edges = cv2.bitwise_or(edges, cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY))

        # Find contours on the combined edges
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours to reduce unnecessary line segments
        approx_contours = [cv2.approxPolyDP(contour, epsilon_factor * cv2.arcLength(contour, True), True) for contour in contours]

        # Merge close contours
        merged_contours = []
        used_contours = set()
        for i, contour1 in enumerate(approx_contours):
            if i in used_contours:
                continue
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            center1 = np.array([x1 + w1/2, y1 + h1/2])
            merged = False
            for j, contour2 in enumerate(approx_contours):
                if i == j or j in used_contours:
                    continue
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                center2 = np.array([x2 + w2/2, y2 + h2/2])
                distance = np.linalg.norm(center1 - center2)
                if distance < distance_threshold:
                    merged_contour = np.vstack((contour1, contour2))
                    merged_contours.append(cv2.convexHull(merged_contour))
                    used_contours.update([i, j])
                    merged = True
                    break
            if not merged:
                merged_contours.append(contour1)

        # Filter out short contours
        filtered_contours = [contour for contour in merged_contours if cv2.arcLength(contour, True) >= min_length]

        # Draw the filtered contours on the original image
        cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

        return filtered_contours