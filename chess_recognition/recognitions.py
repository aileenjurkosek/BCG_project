import numpy as np
import chess
from pathlib import Path
import torch
from PIL import Image
import functools
import cv2
import argparse
import typing
from recap import CfgNode as CN
from timeit import default_timer as timer

from .detect_corners import find_corners, resize_image
from .training import create_occupancy_dataset
from .training import create_piece_dataset
from .utils import Datasets, name_to_piece, DEVICE, device, build_transforms



class ChessRecognizer:

    _squares = list(chess.SQUARES)

    def __init__(self):
        self._corner_detection_cfg = CN.load_yaml_with_base("config/corner_detection.yaml")

        self._occupancy_cfg, self._occupancy_model = self._load_classifier(Path(f"models/occupancy_classifier"))
        self._occupancy_transforms = build_transforms(self._occupancy_cfg, mode=Datasets.TEST)
        self._pieces_cfg, self._pieces_model = self._load_classifier(Path(f"models/piece_classifier"))
        self._pieces_transforms = build_transforms(self._pieces_cfg, mode=Datasets.TEST)
        self._piece_classes = np.array(list(map(name_to_piece, self._pieces_cfg.DATASET.CLASSES)))

    @classmethod
    def _load_classifier(cls, path: Path):
        model_file = next(iter(path.glob("*.pt")))
        yaml_file = next(iter(path.glob("*.yaml")))
        cfg = CN.load_yaml_with_base(yaml_file)
        model = torch.load(model_file, map_location=DEVICE)
        model = device(model)
        model.eval()
        return cfg, model

    def _classify_occupancy(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray) -> np.ndarray:

        warped = create_occupancy_dataset.warp_chessboard_image(img, corners)
        square_imgs = map(functools.partial(create_occupancy_dataset.crop_square, warped, turn=turn), self._squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(axis=-1) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()

        # Display the preliminary results
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        # cv2.imshow('original img: ', img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('warped: ', warped, cv2.COLOR_BGR2RGB)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print("Occupancy: ", occupancy)

        return occupancy

    def _classify_pieces(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray, occupancy: np.ndarray) -> np.ndarray:

        occupied_squares = np.array(self._squares)[occupancy]
        warped = create_piece_dataset.warp_chessboard_image(img, corners)
        piece_imgs = map(functools.partial(create_piece_dataset.crop_square, warped, turn=turn), occupied_squares)
        piece_imgs = map(Image.fromarray, piece_imgs)
        piece_imgs = map(self._pieces_transforms, piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs)
        piece_imgs = device(piece_imgs)
        pieces = self._pieces_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        pieces = self._piece_classes[pieces]
        all_pieces = np.full(len(self._squares), None, dtype=object)
        all_pieces[occupancy] = pieces


        
        # Display the preliminary results
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        # cv2.imshow('original img: ', img)
        # cv2.imshow('warped: ', warped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return all_pieces

    # Predict the positions of all chess pieces on the board
    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        
        with torch.no_grad():
            t1 = timer()
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            t2 = timer()
            occupancy = self._classify_occupancy(img, turn, corners)
            t3 = timer()
            pieces = self._classify_pieces(img, turn, corners, occupancy)
            t4 = timer()

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = f"{corners}/{img_scale}"
            t5 = timer()

            print(f"Corner Detection:         {round(t2-t1,4)} s")
            print(f"Occupancy Classification: {round(t3-t2,4)} s")
            print(f"Piece Classification:     {round(t4-t3,4)} s")
            print(f"Preparing Results:        {round(t5-t4,4)} s\n")

            return board, corners


def main():
    """Main method for chessboard recognition
    
        Args: path to the chessboard image
                --white or --black depending on from which point of view the image was taken
    """

    parser = argparse.ArgumentParser(description="Run the chess recognition pipeline on an input image")
    parser.add_argument("file", help="path to the input image", type=str)
    parser.add_argument("--white", action="store_true", dest="color")
    parser.add_argument("--black", action="store_false", dest="color")
    parser.set_defaults(color=True)
    args = parser.parse_args()

    img = cv2.imread(str(Path(args.file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer()
    board, *_ = recognizer.predict(img, args.color)

    print(board)
    print(f"You can view this position at https://lichess.org/editor/{board.board_fen()}\n")


if __name__ == "__main__":
    main()
