import os
import warnings
from datetime import datetime, timedelta
from logging import getLogger
from typing import Union, Any, Optional

import cv2
import numpy as np
try:
    from jose import jwt
    from passlib.context import CryptContext
except ImportError:
    warnings.warn(
        "Unable to import 'jose' and/or 'passlib.context' modules. "
        "JWT and password hashing will not be available."
    )

from .Log import SERVER_LOGGER_NAME

logger = getLogger(SERVER_LOGGER_NAME)
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
ALGORITHM = "HS256"
JWT_SECRET_KEY = os.environ.get("ML_SERVER_JWT_SECRET_KEY")  # should be kept secret
JWT_REFRESH_SECRET_KEY = os.environ.get("ML_SERVER_JWT_REFRESH_SECRET_KEY")  # should be kept secret
try:
    password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except NameError:
    pass


def get_hashed_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, hashed_pass: str) -> bool:
    return password_context.verify(password, hashed_pass)


def create_token(
    token_type: str, subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    if expires_delta is not None and isinstance(expires_delta, timedelta):
        expires_delta = datetime.utcnow() + expires_delta
    else:
        if token_type == "access":
            expires_delta = datetime.utcnow() + timedelta(
                minutes=ACCESS_TOKEN_EXPIRE_MINUTES
            )
        elif token_type == "refresh":
            expires_delta = datetime.utcnow() + timedelta(
                minutes=REFRESH_TOKEN_EXPIRE_MINUTES
            )

    to_encode = {"exp": expires_delta, "sub": str(subject)}
    return jwt.encode(to_encode, JWT_SECRET_KEY, ALGORITHM)


def resize_cv2_image(img: np.ndarray, resize_w: int):
    """Resize a CV2 (numpy.ndarray) image using ``resize_w``. Width is used and height will be scaled accordingly."""
    lp = "resize:img:"

    if img is not None:
        h, w = img.shape[:2]
        aspect_ratio: float = resize_w / w
        dim: tuple = (resize_w, int(h * aspect_ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        logger.debug(
            f"{lp} success using resize_width={resize_w} - original dimensions: {w}*{h}"
            f" - resized dimensions: {dim[1]}*{dim[0]}",
        )
    else:
        logger.debug(f"{lp} 'resize' called but no image supplied!")
    return img
