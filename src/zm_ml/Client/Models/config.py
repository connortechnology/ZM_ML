import logging
from pathlib import Path
from typing import Dict, List, Tuple, Pattern, Union, Any, Optional

from pydantic import BaseModel, Field, AnyUrl, validator, IPvAnyAddress

from .validators import percentage_and_pixels_validator

logger = logging.getLogger("ZM-ML")


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)


class ZMAPISettings(BaseModel):
    portal: AnyUrl = Field(None)
    api: AnyUrl = Field(None)
    user: str = Field(None)
    password: str = Field(None)
    ssl_verify: bool = Field(True)


class LoggingSettings(BaseModel):
    level: str = Field(logging.INFO)
    console: bool = Field(True)
    integrate_zm: bool = Field(False)
    dir: Path = Field(Path("/var/log/zm"))
    file_name: str = Field(default="zm_ml.log")
    user: str = Field(default="www-data")
    group: str = Field(default="www-data")


class MLAPIRoute(BaseModel):
    name: str = Field(...)
    enabled: bool = Field(True)
    weight: int = Field(0)
    host: Union[IPvAnyAddress, AnyUrl] = Field(...)
    port: int = Field(5000)
    username: str = Field(None)
    password: str = Field(None)
    timeout: int = Field(90)


class MLAPIRoutes(BaseModel):
    routes: List[MLAPIRoute] = Field(default_factory=list)


class MLAPIAnimationSettings(BaseModel):
    class AnimationBaseSettings(BaseModel):
        enabled: bool = Field(False)
        fps: int = Field(ge=0, default=10)
        duration: int = Field(le=120, ge=1, default=10)
        width: int = Field(640)

        def __init__(self, **data: Any):
            logger.debug(f"About to validate '{self.__repr_name__()}'")
            super().__init__(**data)
            logger.debug(f"Validated '{self.__repr_name__()}'")

    class AnimationGIFSettings(AnimationBaseSettings):

        fast: bool = Field(False)

        def __init__(self, **data: Any):
            logger.debug(f"About to validate '{self.__repr_name__()}'")
            super().__init__(**data)
            logger.debug(f"Validated '{self.__repr_name__()}'")

    gif: AnimationGIFSettings = Field(default_factory=AnimationGIFSettings)
    mp4: AnimationBaseSettings = Field(default_factory=AnimationBaseSettings)


class NotificationZMURLOptions(BaseModel):
    mode: str = Field("jpeg")
    scale: int = Field(50)
    max_fps: int = Field(15)
    buffer: int = Field(1000)
    replay: str = Field("single")


class MLNotificationSettings(BaseModel):
    class ZMNinjaNotificationSettings(BaseModel):
        class ZMNinjaFCMSettings(BaseModel):
            enabled: bool = Field(False)
            v1: bool = Field(False)
            local_tokens: Path = Field(
                None,
            )
            replace_messages: bool = Field(False)
            date_fmt: str = Field("%I:%M %p, %d-%b")
            android_priority: str = Field("high")
            log_raw_message: bool = Field(False)
            log_message_id: str = Field(None)
            android_ttl: int = Field(0)

        enabled: bool = Field(True)
        fcm: ZMNinjaFCMSettings = Field(default_factory=ZMNinjaFCMSettings)

    class GotifyNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        host: Union[AnyUrl, IPvAnyAddress, str] = Field("http://localhost:8008")
        token: str = Field(None)
        portal: str = Field(None)
        url_opts: NotificationZMURLOptions = Field(default_factory=NotificationZMURLOptions)

    class PushoverNotificationSettings(BaseModel):
        class SendAnimations(BaseModel):
            enabled: bool = Field(False)
            token: str = Field(None)
            key: str = Field(None)

        enabled: bool = Field(False)
        token: str = Field(None)
        key: str = Field(None)
        portal: str = Field(None)
        animation: SendAnimations = Field(default_factory=SendAnimations)
        url_opts: NotificationZMURLOptions = Field(default_factory=NotificationZMURLOptions)

    class ShellScriptNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        script: str = Field(None)

    class HassNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        host: Union[AnyUrl, IPvAnyAddress] = Field(None)
        token: str = Field(None)
        ssl_verify: bool = Field(True)

    zmninja: ZMNinjaNotificationSettings = Field(default_factory=ZMNinjaNotificationSettings)
    gotify: GotifyNotificationSettings = Field(default_factory=GotifyNotificationSettings)
    pushover: PushoverNotificationSettings = Field(default_factory=PushoverNotificationSettings)
    shell_script: ShellScriptNotificationSettings = Field(default_factory=ShellScriptNotificationSettings)
    hass: HassNotificationSettings = Field(default_factory=HassNotificationSettings)


class APIPullMethod(BaseModel):
    enabled: bool = Field(False)
    fps: int = Field(1)
    attempts: int = Field(3)
    delay: float = Field(1.0)
    check_snapshots: bool = Field(True)
    snapshot_frame_skip: int = Field(3)
    max_frames: int = Field(0)

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")


class DetectionSettings(BaseModel):
    class ImageSettings(BaseModel):
        class PullMethod(BaseModel):
            shm: bool = Field(False)
            api: APIPullMethod = Field(default_factory=APIPullMethod)
            zmu: bool = Field(False)

        class Debug(DefaultEnabled):
            enabled: bool = Field(False)
            path: Path = Field(Path("/tmp"))

        class Annotations(BaseModel):
            class Zones(DefaultEnabled):
                color: Union[str, Tuple[int, int, int]] = Field((255, 0, 0))
                thickness: int = Field(2)

            class Models(DefaultEnabled):
                processor: bool = Field(False)

            zones: Zones = Field(default_factory=Zones)
            models: Models = Field(default_factory=Models)
            confidence: bool = Field(True)

        class Training(DefaultEnabled):
            from tempfile import gettempdir

            enabled: bool = Field(False)
            path: Path = Field(Path(gettempdir()) / "src/training")

        pull_method: PullMethod = Field(default_factory=PullMethod)
        debug: Debug = Field(default_factory=Debug)
        annotation: Annotations = Field(default_factory=Annotations)
        training: Training = Field(default_factory=Training)

    models: Dict = Field(default_factory=dict)
    import_zones: bool = Field(False)
    match_origin_zone: bool = Field(False)
    images: ImageSettings = Field(default_factory=ImageSettings)

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")


class BaseObjectFilters(BaseModel):
    min_conf: float = Field(ge=0.0, le=1.0, default=None)
    total_max_area: Union[float, int, str] = Field(default=None)
    total_min_area: Union[float, int, str] = Field(default=None)
    max_area: Union[float, int, str] = Field(default=None)
    min_area: Union[float, int, str] = Field(default=None)

    # validators
    _normalize_areas = validator(
        "total_max_area", "total_min_area", "max_area", "min_area", allow_reuse=True
    )(percentage_and_pixels_validator)


class ObjectFilters(BaseObjectFilters):
    pattern: Pattern = Field(default=".*")
    labels: Dict[str, BaseObjectFilters] = Field(None)

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")


class FaceFilters(BaseModel):
    pattern: Pattern = Field(default=".*")


class AlprFilters(BaseModel):
    pattern: Pattern = Field(default=".*")
    min_conf: float = Field(ge=0.0, le=1.0, default=0.35)


class StaticObjects(DefaultEnabled):
    enabled: bool = Field(False)
    difference: Union[float, int] = Field(0.1)
    labels: List[str] = Field(default_factory=list)

    _validate_difference = validator("difference", allow_reuse=True)(
        percentage_and_pixels_validator
    )

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")


class MatchFilters(BaseModel):
    object: ObjectFilters = Field(default_factory=ObjectFilters)
    face: FaceFilters = Field(default_factory=FaceFilters)
    alpr: AlprFilters = Field(default_factory=AlprFilters)

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")


class MatchingSettings(BaseModel):
    object_confirm: bool = Field(False)
    static_objects: StaticObjects = Field(default_factory=StaticObjects)
    filters: MatchFilters = Field(default_factory=MatchFilters)


class MonitorZones(BaseModel):
    pattern: Pattern = Field(None)
    points: List = Field(None)
    resolution: str = Field(None)
    static_objects: StaticObjects = Field(None)
    filters: MatchFilters = Field(None)

    @validator("points", pre=True, always=True)
    def validate_points(cls, v, field):
        if v:
            orig = str(v)
            logger.debug(f"Validating '{field.name}' [type: {type(v)}] -> {v}")
            if not isinstance(v, (str, list)):
                raise TypeError(f"Can only be List or string! type={type(v)}")
            elif isinstance(v, str):
                v = [tuple(map(int, x.strip().split(","))) for x in v.split(" ")]
            from shapely.geometry import Polygon

            try:
                Polygon(v)
            except Exception as exc:
                logger.warning(f"Zone points unable to form a valid Polygon: {exc}")
                raise TypeError(
                    f"The polygon points [coordinates] supplied "
                    f"are malformed! -> {orig}"
                )
            else:
                assert isinstance(v, list)
                logger.debug(f"Zone Polygon points are valid -> {v}")

        return v


class MonitorsSettings(BaseModel):
    models: Dict[str, Any] = Field(default_factory=dict)
    zones: Dict[str, MonitorZones] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions:  Dict[str, str] = Field(default_factory=dict)


class ConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    config_path: Path = Field(Path('/etc/zm/ml'))
    zoneminder: ZMAPISettings = Field(default_factory=ZMAPISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    mlapi: MLAPIRoutes = Field(default_factory=MLAPIRoutes)
    animation: MLAPIAnimationSettings = Field(default_factory=MLAPIAnimationSettings)
    notifications: MLNotificationSettings = Field(
        default_factory=MLNotificationSettings
    )
    label_groups: Dict[str, List[str]] = Field(default_factory=dict)
    detection_settings: DetectionSettings = Field(default_factory=DetectionSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)
    monitors: Dict[int, MonitorsSettings] = Field(default_factory=dict)
    end: Optional[Any] = "END"

    @validator("end", always=True)
    def end_of_config(cls, v, values, field, config):
        cfg_path = values["config_path"]
        notification: MLNotificationSettings = values["notifications"]
        if not notification.zmninja.fcm.local_tokens:
            notification.zmninja.fcm.local_tokens = cfg_path / "zmninja_tokens"
    @validator("config_path", always=True, pre=True)
    def val_cfg_path(cls, v):
        if v:
            assert isinstance(v, (Path, str))
            if isinstance(v, str):
                v = Path(v)
        return v

    # @validator("substitutions", pre=True, always=True)
    # def val_subs(cls, v, field, values, config):
    #     testing: Testing = values['testing']
    #     logging.debug(f"Validating {field.name}!! -> {testing = }")
    #     if testing.enabled:
    #         logger.info(f"|----- TESTING IS ENABLED! -----|")
    #         if testing.substitutions:
    #             logger.info(f"Overriding substitutions WITH testing:substitutions")
    #             v = testing.substitutions
    #
    #     return v

    @validator("monitors", always=True)
    def val_mons(cls, v, field, values, config):
        logger.debug(f"validating {field.name} {list(v.keys())}")
        # logger.debug(f"--  {cls = }\n\n--  {v = }\n\n--  {values = }\n\n--  {field = }\n\n--  {config = }")
        return v

    def __init__(self, **data: Any):
        logger.debug(f"About to validate '{self.__repr_name__()}'")
        super().__init__(**data)
        logger.debug(f"Validated '{self.__repr_name__()}'")
