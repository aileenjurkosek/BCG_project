import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_corners(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pattern_size = (8, 8)
    pattern_points = np.zeros((np.prod(pattern_size), 2), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= 20  # Each square is 20mm

    # Define the parameters for goodFeaturesToTrack
    max_corners = 64
    quality_level = 0.001
    min_distance = 20
    block_size = 3
    use_harris_detector = True
    k = 0.04

    # Apply goodFeaturesToTrack to find corners
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance,
                                      None, None, block_size, use_harris_detector, k)

    print("corners shape:", corners.shape)
    print("pattern_points shape:", pattern_points.shape)

    # Find homography matrix using the corner points and the pattern points
    H, _ = cv2.findHomography(corners, pattern_points)

    # Warp the image to remove perspective distortion
    result = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

    return result


if __name__ == "__main__":
    # Load the image
    image = cv2.imread("chessboard.png")

    # Find the corners
    corners = find_corners(image)

    # Show the image with the corners
    fig = plt.figure(num="Corner detection output")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if corners is not None:
       plt.scatter(*corners.T, c="r")
    plt.axis("off")
    plt.show()
