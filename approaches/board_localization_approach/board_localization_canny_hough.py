import cv2
import numpy as np
import matplotlib.pyplot as plt

from coordinates import from_homogenous_coordinates, to_homogenous_coordinates

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
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )


    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find Hough lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # Find intersection points of Hough lines
    intersections = []
    for line1 in lines:
        rho1, theta1 = line1[0]
        for line2 in lines:
            rho2, theta2 = line2[0]
            if abs(theta1 - theta2) > np.pi/4:  # Only consider orthogonal lines
                x, y = np.array([np.cos(theta1), np.sin(theta1)]), np.array([np.cos(theta2), np.sin(theta2)])
                intersection = np.int32(np.round(np.linalg.solve(np.vstack([x, y]), np.array([rho1, rho2]))))
                if intersection[0] >= 0 and intersection[1] >= 0 and intersection[0] < gray.shape[1] and intersection[1] < gray.shape[0]:
                    intersections.append(intersection)

    # Find the four corners
    corner_candidates = np.array(intersections, dtype=np.int32)
    distances = cv2.distanceTransform(255 - edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    corner_scores = distances[corner_candidates[:, 1], corner_candidates[:, 0]]
    corner_ids = np.argsort(corner_scores)[:4]
    corners = corner_candidates[corner_ids]

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
        plt.scatter(*corners.T, c="r")
    plt.axis("off")
    plt.show()
