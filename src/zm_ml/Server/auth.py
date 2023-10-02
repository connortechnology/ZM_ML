from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Annotated, Union, Optional, TYPE_CHECKING, List

import tinydb
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from starlette.requests import Request

from .Log import SERVER_LOGGER_NAME

if TYPE_CHECKING:
    from .Models.config import GlobalConfig, Settings

DFLT_USER_PASSW = "mlzm"
DFLT_USER_UNAME = "zmml"
logger = logging.getLogger(SERVER_LOGGER_NAME)
__all__ = [
    "OAUTH2_SCHEME",
    "pwd_context",
    "UserDB",
    "tinydb",
    "credentials_exception",
    "ZoMiUser",
    "Token",
    "verify_password",
    "get_password_hash",
    "verify_token",
    "create_access_token",
]
credentials_exception: HTTPException = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)
g: Optional[GlobalConfig] = None
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class ZoMiUser(BaseModel):
    """
        DB User Schema

    :param username: ZoMi API Username
    :type username: str
    :param password: Hashed ZoMi API Password
    :type password: str
    :param perms: ZoMi Permissions (WIP)
    :type perms: List[str]
    :param disabled: User Disabled
    :type disabled: bool
    """

    username: str = Field(..., description="ZoMi Username")
    password: Optional[str] = Field(None, description="Hashed ZoMi Password")
    perms: Optional[List[str]] = Field(None, description="ZoMi Permissions (WIP)")
    disabled: Optional[bool] = Field(False, description="User Disabled")


class Token(BaseModel):
    """Represents a JWT token with some extra data"""
    access_token: str
    token_type: str
    # So we dont need to decode the token to get the expire time
    expire: datetime


def check_init(func):
    """Decorator to check if the UserDB is initialized"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert isinstance(self, UserDB), f"self is not a UserDB instance: {self=} -- {type(self)=}"
        assert self.init, f"UserDB not initialized: {self.init=}"
        return func(self, *args, **kwargs)

    return wrapper


def verify_password(
    plain_password: Union[str, bytes], hashed_password: Union[str, bytes]
) -> bool:
    """Verify a password against a bcrypt hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: Union[str, bytes]) -> str:
    """Get a bcrypt hash of a password"""
    return pwd_context.hash(password)


class UserDB:
    """A class to wrap tinydb for user management

    :param db_file: Path to the user database file (Optional)
    """

    db: Optional[tinydb.TinyDB] = None
    table: Optional[tinydb.database.Table] = None
    init: bool = False
    path: Optional[Path] = None

    def __init__(self, db_file: Optional[Union[Path, str]] = None):
        if db_file:
            if isinstance(db_file, str):
                db_file = Path(db_file)
            assert isinstance(db_file, Path), f"Incorrect type! {type(db_file)=}"
            self.set_input(db_file)

    def _get_db(self) -> tinydb.TinyDB:
        return self.db

    def __call__(self, username: str) -> Optional[ZoMiUser]:
        """Default to getting a user by username"""
        return self.get_user(username)

    def disable_user(self, username: str):
        lp = "usr mngmt:disable::"
        # check if the user exists
        if self.table.contains(tinydb.Query().username == username):
            if not self.table.get(tinydb.Query().username == username).get("disabled"):
                logger.debug(f"{lp} User {username} is now disabled")
                self.table.update(
                    {"disabled": True}, tinydb.Query().username == username
                )
                return True
            else:
                logger.warning(f"{lp} User {username} already disabled")
        else:
            logger.warning(f"{lp} User {username} not found")
        return False


    def enable_user(self, username: str):
        lp = "usr mngmt:enable::"
        # check if the user exists
        if self.get_user(username):
            # check if the user is disabled
            if self.table.get(tinydb.Query().username == username).get("disabled"):
                logger.debug(f"{lp} User {username} is now enabled")
                self.table.update(
                    {"disabled": False}, tinydb.Query().username == username
                )
                return True
            else:
                logger.warning(f"{lp} User {username} already enabled")
        else:
            logger.warning(f"{lp} User {username} not found")
        return False

    @staticmethod
    def create_db(db_path: Path) -> bool:
        lp = "user db:create::"
        if db_path.exists():
            logger.warning(f"{lp} DB file {db_path} already exists, using it!")
        else:
            try:
                db_path.touch()
            except Exception as e:
                logger.error(f"{lp} Could not create DB file {db_path}: {e}")
            else:
                logger.info(f"{lp} Created DB file {db_path}")
                return True
        return False

    def _check_add_default_user(self) -> None:
        lp = "user db::"
        if not self.table.all():
            logger.info(
                f"{lp} ANTI-LOCKOUT RULE!!!! No users found, creating default user: "
                f"{DFLT_USER_UNAME} with password: {DFLT_USER_PASSW}\n"
                f"{lp} If you do not want this default user to have access, disable it."
            )

            self.create_user(DFLT_USER_UNAME, DFLT_USER_PASSW, disabled=False)

    def set_input(self, db_path: Path) -> None:
        lp = "user db:set_input::"
        assert isinstance(db_path, Path), f"{lp} db_path is not a Path: {type(db_path)=}"
        if not db_path.exists():
            self.create_db(db_path)
        self.path = db_path
        self.db = tinydb.TinyDB(db_path)
        self.table = self.db.table("users")
        self.init = True
        # check if there are any entries in the DB, if not add a default user
        self._check_add_default_user()

    @check_init
    def create_user(self, username: str, password: str, disabled: bool = False) -> bool:
        lp = "usr mngmt:create::"
        logger.debug(f"{lp} Requested for: {username}")

        # check if the user exists
        if self.table.contains(tinydb.Query().username == username):
            logger.warning(f"{lp} User {username} already exists, use update to change password or use enable/disable")
            return False

        try:
            self.table.insert(
                ZoMiUser(
                    username=username,
                    password=get_password_hash(password),
                    disabled=disabled,
                    perms=["*"],
                ).model_dump()
            )
        except Exception as e:
            logger.error(
                f"{lp} Didnt create user ({username}) ->> Exception [{e.__class__.__name__}]: {e}"
            )
        else:
            logger.info(f"{lp} Created user {username}")
            # remove the default user, if in db
            if self.get_user(DFLT_USER_UNAME):
                logger.info(f"{lp} ANTI LOCKOUT RULE!!! There are other entries in the DB, so there is no need to "
                            f"keep the default user. Removing default user: {DFLT_USER_UNAME}")
                self.delete_user(DFLT_USER_UNAME)

            return True
        return False

    @check_init
    def get_user(self, username: str) -> Optional[ZoMiUser]:
        lp = "usr mngmt:get::"
        user = self.table.get(tinydb.Query().username == username)
        if user:
            logger.debug(f"{lp} Found in DB: {username}")
            try:
                user = ZoMiUser(**user)

            except Exception as e:
                logger.error(f"{lp} Could not create ZoMiUser from DB entry: {e}", exc_info=True)
                raise e
            else:
                if user.disabled:
                    logger.warning(f"{lp} User {username} is disabled")
                    raise HTTPException(status_code=400, detail="User has been marked as disabled")
            return user
        else:
            logger.warning(f"{lp} User {username} not found")
        return None

    @check_init
    def update_user(self, username: str, password: str, disabled: bool = False) -> bool:
        lp = "usr mngmt:update::"
        user = self.get_user(username)
        if user:
            logger.debug(f"{lp} Found in DB: {username}")
            user.password = get_password_hash(password)
            user.disabled = disabled
            self.table.update(user.model_dump(), tinydb.Query().username == username)
            return True
        else:
            logger.warning(f"{lp} User {username} not found")
        return False

    @check_init
    def delete_user(self, username: str):
        lp = "usr mngmt:delete::"
        user = self.get_user(username)
        if user:
            try:
                self.table.remove(tinydb.Query().username == username)
            except Exception as e:
                logger.error(f"{lp} Could not delete user {username}: {e}")
            else:
                logger.debug(f"{lp} Removed: {username}")
                self._check_add_default_user()
                return True

        else:
            logger.warning(f"{lp} User {username} not found")
        return False

    @check_init
    def list_users(self):
        lp = "usr mngmt:list::"
        logger.debug(f"{lp} Listing ALL users")
        return [ZoMiUser(**user) for user in self.table.all()]

    def authenticate_user(
        self,
        username: str,
        password: str,
        cfg: Optional[Settings] = None,
        bypass: bool = False,
    ) -> Optional[Union[ZoMiUser, bool]]:
        global g
        from .app import get_global_config

        lp: str = "usr mngmt:auth::"

        if g is None and cfg is None:
            g = get_global_config()
            cfg = g.config
            logger.debug(f"{lp} Initializing and using global config")
        elif g is not None and cfg is None:
            logger.debug(f"{lp} Using global config")
            cfg = g.config
        else:
            cfg = g.config
            logger.debug(f"{lp} Using global config")

        logger.debug(f"{lp} Checking {username}")

        if not cfg.server.auth.enabled and not bypass:
            logger.warning(
                f"{lp} Authentication disabled in config file, returning bogus user data"
            )
            user = ZoMiUser(
                username=username, password="<BOGUS_DATA>", perms=["*"], disabled=False
            )
        else:
            if self.init:
                user: Optional[ZoMiUser] = self.get_user(username)
                if not user:
                    logger.warning(f"{lp} User: {username} - Does not exist")
                    return False
                if not verify_password(password, user.password):
                    logger.warning(f"{lp} User: {username} - Password is incorrect")
                    return False
            else:
                raise RuntimeError(f"{lp} UserDB not initialized")

        return user

    def __del__(self):
        self.close()

    def close(self):
        if self.db:
            self.db.close()
        self.init = False
    def __repr__(self):
        return f"UserDB({self.path}, active={self.init})"


async def verify_token(
        token: Annotated[str, Depends(OAUTH2_SCHEME)],
        request: Request,
) -> bool:
    """Used as a fastapi Depends() injection to verify a JWT token

    :param token: The JWT token to verify
    :param request: The starlette Request object
    :return: True if the token is valid, False if not
    """
    from .app import get_global_config
    global g
    if g is None:
        g = get_global_config()
    lp = "usr mngmt:token:verify::"
    # logger.debug(f"{lp} Request headers: {request.headers}")
    real_ip = request.client.host
    ret_: bool = False
    try:
        payload = jwt.decode(
            token,
            g.config.server.auth.sign_key.get_secret_value(),
            algorithms=[g.config.server.auth.algorithm],
        )
        username: str = payload.get("sub")
        logger.info(f"{lp} Attempting to verify a request from username: {username} using IP: {real_ip}")
        if username is None:
            raise credentials_exception
    except JWTError as e:
        logger.warning(f"{lp} JWTError: {e}")
    except Exception as e:
        logger.warning(f"{lp} Exception: {e}")
    else:
        user = g.user_db.get_user(username=username)
        if user is None:
            raise credentials_exception
        else:
            logger.debug(f"{lp} User {username} verified!")
            ret_ = True
    return ret_


def create_access_token(data: dict, *args, **kwargs) -> Token:
    """Create a JWT token from a dict of data

    :param data: The data to encode into the token

    :return: A Token object
    """
    from .app import get_global_config

    global g

    if g is None:
        g = get_global_config()

    expires_delta = g.config.server.auth.expire_after
    to_encode = data.copy()
    if expires_delta is not None:
        expire = (datetime.utcnow() + timedelta(minutes=expires_delta)).timestamp()
    else:
        logger.warning(
            f"in create_access_token():: No expires_delta, using default of 15"
        )
        expire = (datetime.utcnow() + timedelta(minutes=15)).timestamp()
    to_encode.update({"exp": str(int(expire))})
    key = g.config.server.auth.sign_key.get_secret_value()
    algo = g.config.server.auth.algorithm

    encoded_jwt = jwt.encode(
        to_encode,
        key,
        algorithm=algo,
    )
    return Token(access_token=encoded_jwt, token_type="bearer", expire=expire)