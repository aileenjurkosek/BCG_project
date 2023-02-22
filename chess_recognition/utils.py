import torch
import functools
import typing
from collections.abc import Iterable
import chess
import numpy as np
from enum import Enum

from recap import CfgNode as CN
import torchvision
#import torchvision
from PIL import Image, ImageOps
from abc import ABC

_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])

#: Device to be used for computation (GPU if available, else CPU).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, torch.nn.Module, typing.List[torch.Tensor],
                 tuple, dict, typing.Generator]


class Datasets(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def device(x: T, dev: str = DEVICE) -> T:
    """Convenience method to move a tensor/module/other structure containing tensors to the device.

    Args:
        x (T): the tensor (or strucure containing tensors)
        dev (str, optional): the device to move the tensor to. Defaults to DEVICE.

    Raises:
        TypeError: if the type was not a compatible tensor

    Returns:
        T: the input tensor moved to the device
    """

    to = functools.partial(device, dev=dev)
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        return x.to(dev)
    elif isinstance(x, list):
        return list(map(to, x))
    elif isinstance(x, tuple):
        return tuple(map(to, x))
    elif isinstance(x, dict):
        return {k: to(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return map(to, x)
    else:
        raise TypeError

def listify(func: typing.Callable[..., typing.Iterable]) -> typing.Callable[..., typing.List]:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper



def color_name(color: chess.Color) -> str:
    """Convert from chess.Color to string

    Args:
        color (chess.Color): 

    Returns:
        str: string of the chess color
    """
    return {chess.WHITE: "white",
            chess.BLACK: "black"}[color]


def get_piece_name(piece: chess.Piece) -> str:
    """Get string of a chess.Piece

    Args:
        piece (chess.Piece): 

    Returns:
        str: 
    """
    return f"{color_name(piece.color)}_{chess.piece_name(piece.piece_type)}"


def name_to_piece(name: str) -> chess.Piece:
    """Get chess.Piece from string

    Args:
        name (str): 

    Returns:
        chess.Piece: 
    """
    color, piece_type = name.split("_")
    color = color == "white"
    piece_type = chess.PIECE_NAMES.index(piece_type)
    return chess.Piece(piece_type, color)


def build_dataset(cfg: CN, mode: Datasets) -> torch.utils.data.Dataset:

    transform = build_transforms(cfg, mode)
    dataset = torchvision.datasets.ImageFolder(root=f"{cfg.DATASET.PATH}/{mode.value}",
                                               transform=transform)
    return dataset


def build_transforms(cfg: CN, mode: Datasets) -> typing.Callable:

    transforms = cfg.DATASET.TRANSFORMS
    t = []
    if transforms.CENTER_CROP:
        t.append(torchvision.transforms.CenterCrop(transforms.CENTER_CROP))
    if mode == Datasets.TRAIN:
        if transforms.RANDOM_HORIZONTAL_FLIP:
            t.append(torchvision.transforms.RandomHorizontalFlip(transforms.RANDOM_HORIZONTAL_FLIP))
        t.append(torchvision.transforms.ColorJitter(brightness=transforms.COLOR_JITTER.BRIGHTNESS,
                               contrast=transforms.COLOR_JITTER.CONTRAST,
                               saturation=transforms.COLOR_JITTER.SATURATION,
                               hue=transforms.COLOR_JITTER.HUE))
        t.append(Shear(transforms.SHEAR))
        t.append(Scale(transforms.SCALE.HORIZONTAL,
                       transforms.SCALE.VERTICAL))
        t.append(Translate(transforms.TRANSLATE.HORIZONTAL,
                           transforms.TRANSLATE.VERTICAL))
    if transforms.RESIZE:
        t.append(torchvision.transforms.Resize(tuple(reversed(transforms.RESIZE))))
    t.extend([torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=_MEAN, std=_STD)])
    return torchvision.transforms.Compose(t)


def build_data_loader(cfg: CN, dataset: torch.utils.data.Dataset, mode: Datasets) -> torch.utils.data.DataLoader:

    shuffle = mode in {Datasets.TRAIN, Datasets.VAL}

    return torch.utils.data.DataLoader(dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=cfg.DATASET.WORKERS)



def to_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:

    return np.concatenate([coordinates,
                           np.ones((*coordinates.shape[:-1], 1))], axis=-1)


def from_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:

    return coordinates[..., :2] / coordinates[..., 2, np.newaxis]


class RecognitionException(Exception):

    def __init__(self, message: str = "unknown error"):
        super().__init__("chess recognition error: " + message)


class ChessboardNotLocatedException(RecognitionException):

    def __init__(self, reason: str = None):
        message = "chessboard could not be located"
        if reason:
            message += ": " + reason
        super().__init__(message)


class _HVTransform(ABC):

    def __init__(self, horizontal: typing.Union[float, tuple, None], vertical: typing.Union[float, tuple, None]):
        self.horizontal = self._get_tuple(horizontal)
        self.vertical = self._get_tuple(vertical)

    _default_value = None

    @classmethod
    def _get_tuple(cls, value: typing.Union[float, tuple, None]) -> tuple:
        if value is None:
            return cls._default_value, cls._default_value
        elif isinstance(value, (tuple, list)):
            return tuple(map(float, value))
        elif isinstance(value, (float, int)):
            return tuple(map(float, (value, value)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.horizontal}, {self.vertical})"


class Scale(_HVTransform):

    _default_value = 1.

    def __call__(self, img: Image) -> Image:
        w, h = img.size
        w_scale = np.random.uniform(*self.horizontal)
        h_scale = np.random.uniform(*self.vertical)
        w_, h_ = map(int, (w*w_scale, h*h_scale))
        img = img.resize((w_, h_))
        img = img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, h_-h))
        return img


class Shear:

    def __init__(self, amount: typing.Union[tuple, float, int, None]):
        self.amount = amount

    @classmethod
    def _shear(cls, img: Image, amount: float) -> Image:
        img = ImageOps.flip(img)
        img = img.transform(img.size, Image.AFFINE,
                            (1, -amount, 0, 0, 1, 0))
        img = ImageOps.flip(img)
        return img

    def __call__(self, img: Image) -> Image:
        if not self.amount:
            return img
        if isinstance(self.amount, (tuple, list)):
            min_val, max_val = sorted(self.amount)
        else:
            min_val = max_val = self.amount

        amount = np.random.uniform(low=min_val, high=max_val)
        return self._shear(img, amount)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.amount})"


class Translate(_HVTransform):

    _default_value = 0.

    def __call__(self, img: Image) -> Image:
        w, h = img.size
        w_translate = np.random.uniform(*self.horizontal)
        h_translate = np.random.uniform(*self.vertical)
        w_, h_ = map(int, (w*w_translate, h*h_translate))
        img = img.transform((w, h), Image.AFFINE, (1, 0, -w_, 0, 1, h_))
        return img


def unnormalize(x: typing.Union[torch.Tensor, np.ndarray]) -> typing.Union[torch.Tensor, np.ndarray]:

    return x * _STD + _MEAN