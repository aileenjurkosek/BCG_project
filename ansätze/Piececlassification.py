import cv2
import numpy as np


def piece_classification(img):
    pieces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 15:  # Spielsteine erkennen
            piece = img[y:y + h, x:x + w]
            pieces.append(piece)

    # Spielsteine klassifizieren
    classified_pieces = []
    for piece in pieces:
        # Farb-Histogramm erstellen
        piece_hsv = cv2.cvtColor(piece, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([piece_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Klassifizieren basierend auf Farb-Histogramm
        if hist[0] > 0.5:
            classified_pieces.append('black')
        else:
            classified_pieces.append('white')

    return classified_pieces