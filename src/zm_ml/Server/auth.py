import logging
from datetime import datetime, timedelta
from typing import Annotated, Union, Optional, Any, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import tinydb

from .Log import SERVER_LOGGER_NAME
from .Models.config import ZoMiUser

logger = logging.getLogger(SERVER_LOGGER_NAME)
__all__ = [
    "OAUTH2_SCHEME",
    "SECRET_KEY",
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "pwd_context",
    "User",
]
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="login")
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 90
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
User = ZoMiUser

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        # "password": "admin",
        "password": "$2b$12$XceLz2D.rP/xTrbx2AOsQe0WJH4zt8cWrqxIAVOxgpTB01lodA5.q",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


def verify_password(
    plain_password: Union[str, bytes], hashed_password: Union[str, bytes]
) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: Union[str, bytes]) -> str:
    return pwd_context.hash(password)


def get_user(db: Dict[str, Any], username: str) -> Optional[ZoMiUser]:
    if username in db:
        logger.debug(f"in get_user():: User {username} found in db")
        user_dict = db[username]
        return ZoMiUser(**user_dict)
    else:
        logger.debug(f"in get_user():: User {username} not found in db")
        return None


def authenticate_user(db_entry, username: str, password: str):
    logger.debug(f"in auth_user():: Authenticating user {username}")

    user = get_user(db_entry, username)
    if not user:
        logger.warning(f"in auth_user():: User {username} not found")
        return False
    if not verify_password(password, user.hashed_password):
        logger.warning(f"in auth_user():: Password for user {username} is incorrect")
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(OAUTH2_SCHEME)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
