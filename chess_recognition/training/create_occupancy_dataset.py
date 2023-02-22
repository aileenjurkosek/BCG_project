from pathlib import Path
import cv2
from PIL import Image
import json
import numpy as np
import chess
import os
import shutil
import argparse

RENDERS_DIR = Path("data")
OUT_DIR = Path("data/occupancy")
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE


def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    """Crops a chess square from the warped input image for occupancy classsification

    Args:
        img (np.ndarray): the warped input image
        square (chess.Square): the square to crop
        turn (chess.Color): the current player

    Returns:
        np.ndarray: the cropped square image
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warps the chessboard image so the squares align on a grid

    Args:
        img (np.ndarray): original image
        corners (np.ndarray): corner points of the chessboard

    Returns:
        np.ndarray: the warped image
    """
    src_points = sort_corner_points(corners)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=float)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))


def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Sort the detected corner points clockwise beginning from top left

    Args:
        points (np.ndarray): unsorted corner points

    Returns:
        np.ndarray: sorted corner points
    """
    # First, order by y-coordinate
    points = points[points[:, 1].argsort()]
    # Sort top x-coordinates
    points[:2] = points[:2][points[:2, 0].argsort()]
    # Sort bottom x-coordinates (reversed)
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]

    return points


def _extract_squares_from_sample(id: str, subset: str = "", input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR):
    img = cv2.imread(str(input_dir / subset / (id + ".png")))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with (input_dir / subset / (id + ".json")).open("r") as f:
        label = json.load(f)

    corners = np.array(label["corners"], dtype=float)
    unwarped = warp_chessboard_image(img, corners)

    board = chess.Board(label["fen"])

    for square in chess.SQUARES:
        target_class = "empty" if board.piece_at(
            square) is None else "occupied"
        piece_img = crop_square(unwarped, square, label["white_turn"])
        with Image.fromarray(piece_img, "RGB") as piece_img:
            piece_img.save(output_dir / subset / target_class /
                           f"{id}_{chess.square_name(square)}.png")


def create_dataset(input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR):
    """ Create the dataset for occupancy classification training

    Args:
        input_dir (Path, optional): Path to the unmodified dataset. Defaults to RENDERS_DIR.
        output_dir (Path, optional): Path to the created occupancy classification dataset. Defaults to OUT_DIR.
    """
    for subset in ("train", "val", "test"):
        for c in ("empty", "occupied"):
            folder = output_dir / subset / c
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
        samples = list((input_dir / subset).glob("*.png"))
        for i, img_file in enumerate(samples):
            if len(samples) > 100 and i % int(len(samples) / 100) == 0:
                print(f"{i / len(samples)*100:.0f}%")
            _extract_squares_from_sample(img_file.stem, subset,
                                         input_dir, output_dir)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Create the dataset for occupancy classification.").parse_args()
    print("Creating dataset")
    create_dataset()
