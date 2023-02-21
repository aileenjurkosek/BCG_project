
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_corners(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the chessboard pattern
    pattern_size = (8, 8)
    pattern_points = np.zeros((np.prod(pattern_size), 2), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= 20  # Each square is 20mm

    # Define the parameters for goodFeaturesToTrack
    max_corners = 200
    quality_level = 0.001
    min_distance = 20
    block_size = 3
    use_harris_detector = True
    k = 0.04

    # Apply goodFeaturesToTrack to find corners
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance,
                                      None, None, block_size, use_harris_detector, k)


    # Refine the corner positions
    corners = np.array(corners, dtype=np.float32)
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    )

    print("corners shape:", corners.shape)
    print("pattern_points shape:", pattern_points.shape)
    return corners


if __name__ == "__main__":
    # Load the image
    image = cv2.imread("chessboard.png")

    # Find the corners
    corners = find_corners(image)

    # Show the image with the corners
    fig = plt.figure(num="Corner detection output")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if corners is not None:
        plt.scatter(corners[:, 0, 0], corners[:, 0, 1], c="r", s=10)
    plt.axis("off")
    plt.show()
