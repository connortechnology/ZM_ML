import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Pattern, Union, Any, Optional

from pydantic import BaseModel, Field, AnyUrl, validator, IPvAnyAddress, SecretStr

from .validators import percentage_and_pixels_validator, no_scheme_url_validator

logger = logging.getLogger("ML-Client")


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)


class ZMAPISettings(BaseModel):
    portal: AnyUrl = Field(None)
    api: AnyUrl = Field(None)
    user: str = Field(None)
    password: SecretStr = Field(None)
    ssl_verify: bool = Field(True)

    # validators
    _validate_host_portal = validator("api", "portal", allow_reuse=True, pre=True)(
        no_scheme_url_validator
    )


class LoggingSettings(BaseModel):
    level: str = Field(logging.INFO)
    console: bool = Field(True)
    integrate_zm: bool = Field(False)
    log_to_file: bool = Field(False)
    dir: Path = Field(Path("/var/log/zm"))
    file_name: str = Field(default="zm_ml.log")
    user: str = Field(default="www-data")
    group: str = Field(default="www-data")


class MLAPIRoute(BaseModel):
    name: str = Field(...)
    enabled: bool = Field(True)
    weight: int = Field(0)
    host: AnyUrl = Field(...)
    port: int = Field(5000)
    username: str = Field(None)
    password: SecretStr = Field(None)
    timeout: int = Field(90)

    # validators
    _validate_host_portal = validator("host", allow_reuse=True, pre=True)(
        no_scheme_url_validator
    )


class MLAPIRoutes(BaseModel):
    routes: List[MLAPIRoute] = Field(default_factory=list)


class MLAPIAnimationSettings(BaseModel):
    class AnimationBaseSettings(BaseModel):
        enabled: bool = Field(False)
        fps: int = Field(ge=0, default=10)
        duration: int = Field(le=120, ge=1, default=10)
        width: int = Field(640)

    class AnimationGIFSettings(AnimationBaseSettings):
        fast: bool = Field(False)

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
            local_tokens: Path = Field(None)
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
        host: AnyUrl = Field(None)
        token: str = Field(None)
        portal: AnyUrl = Field(None)
        url_opts: NotificationZMURLOptions = Field(
            default_factory=NotificationZMURLOptions
        )

        # validators
        _validate_host_portal = validator("host", "portal", allow_reuse=True, pre=True)(
            no_scheme_url_validator
        )

    class PushoverNotificationSettings(BaseModel):
        class SendAnimations(BaseModel):
            enabled: bool = Field(False)
            token: str = Field(None)
            key: str = Field(None)

        enabled: bool = Field(False)
        token: str = Field(None)
        key: str = Field(None)
        portal: Union[IPvAnyAddress, AnyUrl] = Field(None)
        animation: SendAnimations = Field(default_factory=SendAnimations)
        url_opts: NotificationZMURLOptions = Field(
            default_factory=NotificationZMURLOptions
        )

        # validators
        _validate_host_portal = validator("portal", allow_reuse=True, pre=True)(
            no_scheme_url_validator
        )

    class ShellScriptNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        script: str = Field(None)

    class HassNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        host: AnyUrl = Field(None)
        token: str = Field(None)
        ssl_verify: bool = Field(True)

        # validators
        _validate_host_portal = validator("host", allow_reuse=True, pre=True)(
            no_scheme_url_validator
        )

    zmninja: ZMNinjaNotificationSettings = Field(
        default_factory=ZMNinjaNotificationSettings
    )
    gotify: GotifyNotificationSettings = Field(
        default_factory=GotifyNotificationSettings
    )
    pushover: PushoverNotificationSettings = Field(
        default_factory=PushoverNotificationSettings
    )
    shell_script: ShellScriptNotificationSettings = Field(
        default_factory=ShellScriptNotificationSettings
    )
    hass: HassNotificationSettings = Field(default_factory=HassNotificationSettings)


class APIPullMethod(BaseModel):
    enabled: bool = Field(False)
    fps: int = Field(1)
    attempts: int = Field(3)
    delay: float = Field(1.0)
    check_snapshots: bool = Field(True)
    snapshot_frame_skip: int = Field(3)
    max_frames: int = Field(0)


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


class BaseObjectFilters(BaseModel):
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    total_max_area: Union[float, int, str, None] = Field(default=None)
    total_min_area: Union[float, int, str, None] = Field(default=None)
    max_area: Union[float, int, str, None] = Field(default=None)
    min_area: Union[float, int, str, None] = Field(default=None)

    # validators
    _normalize_areas = validator(
        "total_max_area", "total_min_area", "max_area", "min_area", allow_reuse=True
    )(percentage_and_pixels_validator)


class OverRideObjectFilters(BaseObjectFilters):
    pattern: Optional[Pattern] = None
    labels: Optional[
        Dict[str, Union[BaseObjectFilters, "OverRideObjectFilters", Dict]]
    ] = None

    # _normalize_areas = validator(
    #     "total_max_area", "total_min_area", "max_area", "min_area", allow_reuse=True, pre=True, always=True
    # )(percentage_and_pixels_validator)


class ObjectFilters(BaseObjectFilters):
    pattern: Optional[Pattern] = None
    labels: Optional[
        Dict[str, Union[BaseObjectFilters, OverRideObjectFilters]]
    ] = Field(None)


class FaceFilters(BaseModel):
    pattern: Optional[Pattern] = Field(default=re.compile("(.*)"))


class OverRideFaceFilters(BaseModel):
    pattern: Optional[Pattern] = None


class AlprFilters(BaseModel):
    pattern: Optional[Pattern] = Field(default=re.compile("(.*)"))
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=0.35)


class OverRideAlprFilters(BaseModel):
    pattern: Optional[Pattern] = None
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=None)


class StaticObjects(DefaultEnabled):
    enabled: Optional[bool] = Field(False)
    difference: Optional[Union[float, int]] = Field(0.1)
    labels: List[str] = Field(default_factory=list)
    ignore_labels: List[str] = Field(default_factory=list)

    _validate_difference = validator("difference", allow_reuse=True)(
        percentage_and_pixels_validator
    )


class OverRideStaticObjects(BaseModel):
    enabled: bool = None
    difference: Optional[Union[float, int]] = None
    labels: Optional[List[str]] = Field(None)
    ignore_labels: Optional[List[str]] = Field(None)

    _validate_difference = validator(
        "difference", allow_reuse=True, pre=True, always=True
    )(percentage_and_pixels_validator)


class MatchFilters(BaseModel):
    object: ObjectFilters = Field(default_factory=ObjectFilters)
    face: FaceFilters = Field(default_factory=FaceFilters)
    alpr: AlprFilters = Field(default_factory=AlprFilters)


class OverRideMatchFilters(BaseModel):
    object: OverRideObjectFilters = Field(default_factory=OverRideObjectFilters)
    face: OverRideFaceFilters = Field(default_factory=OverRideFaceFilters)
    alpr: OverRideAlprFilters = Field(default_factory=OverRideAlprFilters)


class MatchingSettings(BaseModel):
    object_confirm: bool = Field(False)
    static_objects: StaticObjects = Field(default_factory=StaticObjects)
    filters: MatchFilters = Field(default_factory=MatchFilters)


class MonitorZones(BaseModel):
    enabled: bool = Field(True)
    points: List[Tuple[int, int]] = None
    resolution: Optional[Tuple[int, int]] = None
    object_confirm: Optional[bool] = None
    static_objects: Union[OverRideStaticObjects, StaticObjects] = Field(
        default_factory=OverRideStaticObjects
    )
    filters: Union[MatchFilters, OverRideMatchFilters] = Field(
        default_factory=OverRideMatchFilters
    )

    _RESOLUTION_STRINGS: Dict[str, Tuple[int, int]] = {
        # pixel resolution string to tuple, feed it .casefold().strip()'d string's
        "4kuhd": (3840, 2160),
        "uhd": (3840, 2160),
        "4k": (4096, 2160),
        "6MP": (3072, 2048),
        "5MP": (2592, 1944),
        "4MP": (2688, 1520),
        "3MP": (2048, 1536),
        "2MP": (1600, 1200),
        "1MP": (1280, 1024),
        "1440p": (2560, 1440),
        "2k": (2048, 1080),
        "1080p": (1920, 1080),
        "960p": (1280, 960),
        "720p": (1280, 720),
        "fullpal": (720, 576),
        "fullntsc": (720, 480),
        "pal": (704, 576),
        "ntsc": (704, 480),
        "4cif": (704, 480),
        "2cif": (704, 240),
        "cif": (352, 240),
        "qcif": (176, 120),
        "480p": (854, 480),
        "360p": (640, 360),
        "240p": (426, 240),
        "144p": (256, 144),
    }

    @validator("resolution", pre=True, always=True)
    def _validate_resolution(cls, v):
        logger.debug(f"Validating Monitor Zone resolution: {v}")
        if not v:
            logger.warning("No resolution provided for monitor zone, will not be able to scale "
                           "zone Polygon if resolution changes")
        elif isinstance(v, str):
            v = v.casefold().strip()
            if v in cls._RESOLUTION_STRINGS:
                v = cls._RESOLUTION_STRINGS[v]
            elif v not in cls._RESOLUTION_STRINGS:
                # check for a valid resolution string
                import re
                # WxH
                if re.match(r"^\d+x\d+$", v):
                    v = tuple(int(x) for x in v.split("x"))
                # W*H
                elif re.match(r"^\d+\*\d+$", v):
                    v = tuple(int(x) for x in v.split("*"))
                # W,H
                elif re.match(r"^\d+,\d+$", v):
                    v = tuple(int(x) for x in v.split(","))
                else:
                    logger.warning(
                        f"Invalid resolution string: {v}. Valid strings are: W*H WxH W,H OR {', '.join(cls._RESOLUTION_STRINGS)}"
                    )
        logger.debug(f"Validated Monitor Zone resolution: {v}")
        return v

    @validator("points", pre=True, always=True)
    def validate_points(cls, v, field):
        if v:
            orig = str(v)
            if not isinstance(v, (str, list)):
                raise TypeError(
                    f"'{field.name}' Can only be List or string! type={type(v)}"
                )
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

        return v


class MonitorsSettings(BaseModel):
    models: Optional[Dict[str, Any]] = Field(default_factory=dict)
    object_confirm: Optional[bool] = None
    static_objects: Optional[OverRideStaticObjects] = Field(default_factory=OverRideStaticObjects)
    filters: Optional[OverRideMatchFilters] = Field(default_factory=OverRideMatchFilters)
    zones: Optional[Dict[str, MonitorZones]] = Field(default_factory=dict)


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions: Dict[str, str] = Field(default_factory=dict)


class SystemSettings(BaseModel):
    variable_data_path: Path = Field(Path("/var/lib/zm_ml"))


class ConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    config_path: Path = Field(Path("/etc/zm/ml"))
    system: SystemSettings = Field(default_factory=SystemSettings)
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

    @validator("config_path", always=True, pre=True)
    def val_cfg_path(cls, v):
        if v:
            assert isinstance(v, (Path, str))
            if isinstance(v, str):
                v = Path(v)
        return v
