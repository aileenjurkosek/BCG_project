import cv2
import numpy as np

# Load image
image = cv2.imread("chessboard.png")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect chessboard corners
found, corners = cv2.findChessboardCorners(gray, (7,7), None)

# Draw chessboard corners on the image if found
if found:
    cv2.drawChessboardCorners(image, (7,7), corners, found)

# Display the image with chessboard corners
cv2.imshow("Chessboard corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()