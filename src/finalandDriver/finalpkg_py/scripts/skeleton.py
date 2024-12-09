import cv2
import numpy as np

def skeletonize_image(image):
    """Convert a binary image to its skeletonized form."""
    skeleton = np.zeros(image.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        # Morphological operations
        open_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, open_img)
        eroded = cv2.erode(image, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        if cv2.countNonZero(image) == 0:
            break

    return skeleton

def find_and_approximate_contours(image):
    """Find contours in the image and approximate them."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approximations = [cv2.approxPolyDP(cnt, epsilon=0.001*cv2.arcLength(cnt, True), closed=False) for cnt in contours]
    return approximations

# Load the image
image = cv2.imread('images/rob.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to get a binary image
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Skeletonize the binary image
skeleton = skeletonize_image(binary_image)
print("print skeletonized")
# Find and approximate contours
contours = find_and_approximate_contours(skeleton)

# Visualize the results
output_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for contour in contours:
    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 1)

# Show the images
cv2.imshow('Skeleton', skeleton)
cv2.imshow('Contours', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
