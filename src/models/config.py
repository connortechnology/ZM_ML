import logging
from pathlib import Path
from typing import Dict, List, Tuple, Pattern, Union, Any

from pydantic import BaseModel, Field, AnyUrl, validator

from validators import percentage_and_pixels_validator

logger = logging.getLogger("src")


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)


class ZMAPISettings(BaseModel):
    portal: AnyUrl = Field(None)
    api: AnyUrl = Field(None)
    user: str = Field(None)
    password: str = Field(None)
    ssl_verify: bool = Field(None)


class LoggingSettings(BaseModel):
    level: str = Field(logging.INFO)
    console: bool = Field(True)
    integrate_zm: bool = Field(False)
    dir: Path = Field(Path("/var/log/zm"))
    file_name: str = Field(default="src.log")
    user: str = Field(default="www-data")
    group: str = Field(default="www-data")

class MLAPIRoute(BaseModel):
    name: str = Field(...)
    enabled: bool = Field(True)
    weight: int = Field(0)
    host: AnyUrl = Field(...)
    port: int = Field(...)
    username: str = Field(None)
    password: str = Field(None)


class MLAPIRoutes(BaseModel):
    routes: List[MLAPIRoute] = Field(default_factory=list)


class MLAPIAnimationSettings(BaseModel):
    class AnimationGIFSettings(BaseModel):
        enabled: bool = Field(False)
        fps: int = Field(10)
        duration: int = Field(10)
        width: int = Field(640)
        fast: bool = Field(False)

    class AnimationMP4Settings(BaseModel):
        enabled: bool = Field(False)
        fps: int = Field(10)
        duration: int = Field(10)
        width: int = Field(640)

    gif: AnimationGIFSettings = Field(default="gif")
    mp4: AnimationMP4Settings = Field(default="mp4")


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
                Path("/etc/zm/mlapi/zmninja_tokens.txt"),
                env="ZMNINJA_NOTIFICATION_FCM_TOKENS",
            )
            replace_messages: bool = Field(
                False
            )
            date_fmt: str = Field(
                "%I:%M %p, %d-%b"
            )
            android_priority: str = Field(
                "high"
            )
            log_raw_message: bool = Field(
                False
            )
            log_message_id: str = Field(
                default_factory=str
            )
            android_ttl: int = Field(0)

        enabled: bool = Field(True)
        fcm: ZMNinjaFCMSettings = Field(default="fcm")

    class GotifyNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        host: str = Field("http://localhost:8008")
        token: str = Field(default_factory=str)
        portal: str = Field(default_factory=str)
        url_opts: NotificationZMURLOptions = Field(
            default=NotificationZMURLOptions
        )

    class PushoverNotificationSettings(BaseModel):
        class SendAnimations(BaseModel):
            enabled: bool = Field(
                False
            )
            token: str = Field(
                default_factory=str
            )
            key: str = Field(
                default_factory=str
            )

        enabled: bool = Field(False)
        token: str = Field(default_factory=str)
        key: str = Field(default_factory=str)
        portal: str = Field(default_factory=str)
        animation: SendAnimations = Field(
            default_factory=SendAnimations
        )
        url_opts: NotificationZMURLOptions = Field(
            default=NotificationZMURLOptions
        )

    class ShellScriptNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        script: str = Field(
            default_factory=str
        )

    class HassNotificationSettings(BaseModel):
        enabled: bool = Field(False)
        host: str = Field(default_factory=str)
        token: str = Field(default_factory=str)
        ssl_verify: bool = Field(True)

    zmninja: ZMNinjaNotificationSettings = Field(
        default="zmninja"
    )
    gotify: GotifyNotificationSettings = Field(
        default="gotify"
    )
    pushover: PushoverNotificationSettings = Field(
        default="pushover"
    )
    shell_script: ShellScriptNotificationSettings = Field(
        default="shell_script"
    )
    hass: HassNotificationSettings = Field(
        HassNotificationSettings
    )


class APIPullMethod(BaseModel):
    enabled: bool = Field(False)
    fps: int = Field(1)
    attempts: int = Field(3)
    attempt_delay: float = Field(0.5)
    delay: float = Field(0.0)
    check_snapshots: bool = Field(False)
    snapshot_frame_skip: int = Field(3)



class DetectionSettings(BaseModel):
    class ImageSettings(BaseModel):
        class PullMethod(BaseModel):
            shm: bool = Field(False)
            api: APIPullMethod = Field(default=APIPullMethod)
            zmu: bool = Field(False)

        class Debug(DefaultEnabled):
            enabled: bool = Field(False)
            path: Path = Field(Path("/tmp"))

        class Annotations(BaseModel):
            class Zones(DefaultEnabled):
                color: Tuple[int, int, int] = Field(
                    (255, 0, 0)
                )
                thickness: int = Field(2)

            class Models(DefaultEnabled):
                processor: bool = Field(False)

            zones: Zones = Field(default=Zones)
            models: Models = Field(default=Models)
            confidence: bool = Field(True)

        class Training(DefaultEnabled):
            from tempfile import gettempdir

            enabled: bool = Field(False)
            path: Path = Field(
                Path(gettempdir()) / "src/training"
            )
        pull_method: PullMethod = Field(default=PullMethod)
        debug: Debug = Field(default=Debug)
        annotation: Annotations = Field(default=Annotations)
        training: Training = Field(default=Training)

    models: Dict = Field(default_factory=dict)
    import_zones: bool = Field(False)
    match_origin_zone: bool = Field(False)
    images: ImageSettings = Field(ImageSettings)


class BaseObjectFilters(BaseModel):
    min_conf: float = Field(ge=0.0, le=1.0, default=0.35)
    total_max_area: Union[float, int] = Field(default=1.0)
    total_min_area: Union[float, int] = Field(default='1px')
    max_area: Union[float, int] = Field(default=1.0)
    min_area: Union[float, int] = Field(default='1px')

    # validators
    _normalize_areas = validator('total_max_area', 'total_min_area', 'max_area', 'min_area', allow_reuse=True)\
        (percentage_and_pixels_validator)


class ObjectFilters(BaseObjectFilters):
    pattern: Pattern = Field(default=".*")
    labels: Dict[str, BaseObjectFilters] = Field(None)


class FaceFilters(BaseModel):
    pattern: Pattern = Field(default=".*")


class AlprFilters(BaseModel):
    pattern: Pattern = Field(default=".*")
    min_conf: float = Field(ge=0.0, le=1.0, default=0.35)


class StaticObjects(DefaultEnabled):
    enabled: bool = Field(False)
    difference: Union[float, int] = Field(0.1)
    labels: List[str] = Field(
        default_factory=list
    )

    _validate_difference = validator('difference', allow_reuse=True)\
        (percentage_and_pixels_validator)


class MatchFilters(BaseModel):
    object: ObjectFilters = Field(default=ObjectFilters)
    face: FaceFilters = Field(default=FaceFilters)
    alpr: AlprFilters = Field(default=AlprFilters)


class MatchingSettings(BaseModel):
    multi_confirm: bool = Field(False)
    static_objects: StaticObjects = Field(
        default=StaticObjects
    )
    filters: MatchFilters = Field(default=MatchFilters)


class MonitorZones(BaseModel):
    pattern: Pattern = Field(None)
    points: List[Tuple[int, int]] = Field(None)
    resolution: str = Field(None)
    static_objects: StaticObjects = Field(None)
    filters: MatchFilters = Field(None)


class MonitorsSettings(BaseModel):
    models: Dict[str, Any] = Field(default_factory=list)
    zones: Dict[str, MonitorZones] = Field(default_factory=dict)


class ConfigFileModel(BaseModel):
    substitutions: Dict[str, str] = Field(default_factory=dict)
    zoneminder: ZMAPISettings = Field(default=ZMAPISettings)
    logging: LoggingSettings = Field(default=LoggingSettings)
    mlapi: MLAPIRoutes = Field(default=MLAPIRoutes)
    animation: MLAPIAnimationSettings = Field(default=MLAPIAnimationSettings)
    notifications: MLNotificationSettings = Field(default=MLNotificationSettings)
    label_groups: Dict[str, List[str]] = Field(default_factory=dict)
    detection_settings: DetectionSettings = Field(default=DetectionSettings)
    matching: MatchingSettings = Field(default=MatchingSettings)
    monitors: Dict[int, MonitorsSettings] = Field(default_factory=dict)
