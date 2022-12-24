import logging
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Pattern, Union, Any, Optional

from pydantic import BaseModel, Field, AnyUrl, validator, IPvAnyAddress, SecretStr

from .validators import validate_percentage_or_pixels, validate_no_scheme_url, str_to_path, validate_log_level

logger = logging.getLogger("ZM_ML-Client")


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)


class DefaultNotEnabled(DefaultEnabled):
    enabled: bool = Field(False)


class ZMAPISettings(BaseModel):
    class ZMMisc(BaseModel):
        write_notes: bool = Field(True)
    misc: ZMMisc = Field(ZMMisc())
    portal: AnyUrl = Field(None)
    api: AnyUrl = Field(None)
    user: Optional[SecretStr] = Field(None)
    password: Optional[SecretStr] = Field(None)
    ssl_verify: bool = Field(True)

    # validators
    _validate_host_portal = validator("api", "portal", allow_reuse=True, pre=True)(
        validate_no_scheme_url
    )


class LoggingLevelBase(BaseModel):
    level: Optional[int] = None

    _validate_log_level = validator('level', allow_reuse=True, pre=True, always=True)(validate_log_level)


class LoggingSettings(LoggingLevelBase):
    class ConsoleLogging(DefaultEnabled, LoggingLevelBase):
        pass

    class SyslogLogging(DefaultNotEnabled, LoggingLevelBase):
        address: Optional[str] = Field("")

    class FileLogging(DefaultEnabled, LoggingLevelBase):
        path: Path = Field('/var/log/zm')
        filename_prefix: str = Field("zmML")
        user: str = Field(default="www-data")
        group: str = Field(default="www-data")

        _validate_path = validator("path", allow_reuse=True, pre=True)(
            str_to_path
        )
    class SanitizeLogging(DefaultNotEnabled):
        replacement_str: str = Field(default="<sanitized>")

    class IntegrateZMLogging(DefaultNotEnabled):
        debug_level: int = Field(default=4)

    level = logging.INFO
    console: ConsoleLogging = Field(default_factory=ConsoleLogging)
    syslog: SyslogLogging = Field(default_factory=SyslogLogging)
    integrate_zm: IntegrateZMLogging = Field(default_factory=IntegrateZMLogging)
    file: FileLogging = Field(default_factory=FileLogging)
    sanitize: SanitizeLogging = Field(default_factory=SanitizeLogging)


class ServerRoute(BaseModel):
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
        validate_no_scheme_url
    )


class ServerRoutes(BaseModel):
    routes: List[ServerRoute] = Field(default_factory=list)


class AnimationSettings(BaseModel):
    class AnimationBaseSettings(BaseModel):
        enabled: bool = Field(False)

    class AnimationGIFSettings(AnimationBaseSettings):
        fast: bool = Field(False)

    gif: AnimationGIFSettings = Field(default_factory=AnimationGIFSettings)
    mp4: AnimationBaseSettings = Field(default_factory=AnimationBaseSettings)
    width: int = Field(640)

    low_memory: bool = Field(False)
    overwrite: bool = Field(False)
    max_attempts: int = Field(ge=1, default=3)
    attempt_delay: float = Field(ge=0.1, default=2.9, description="Delay between attempts in seconds")


class NotificationZMURLOptions(BaseModel):
    mode: str = Field("jpeg")
    scale: int = Field(50)
    max_fps: int = Field(15)
    buffer: int = Field(1000)
    replay: str = Field("single")


class MLNotificationSettings(BaseModel):
    class ZMNinjaNotificationSettings(BaseModel):
        class ZMNinjaFCMSettings(BaseModel):
            class FCMV1Settings(BaseModel):
                enabled: bool = Field(True)
                key: Optional[SecretStr] = None
                url: Optional[AnyUrl] = None

            v1: FCMV1Settings = Field(default_factory=FCMV1Settings)
            token_file: Path = Field(None)
            replace_messages: bool = Field(False)
            date_fmt: str = Field("%I:%M %p, %d-%b")
            android_priority: str = Field("high")
            log_raw_message: bool = Field(False)
            log_message_id: str = Field(None)
            android_ttl: int = Field(0)

        enabled: bool = Field(True)
        fcm: ZMNinjaFCMSettings = Field(default_factory=ZMNinjaFCMSettings)

    class GotifyNotificationSettings(BaseModel):
        test_image: bool = Field(False)
        enabled: bool = Field(False)
        host: AnyUrl = Field(None)
        token: str = Field(None)
        portal: AnyUrl = Field(None)
        link_url: bool = Field(False)
        link_user: Optional[SecretStr] = Field(None)
        link_pass: Optional[SecretStr] = Field(None)
        _push_auth: Optional[SecretStr] = Field(None)

        url_opts: NotificationZMURLOptions = Field(
            default_factory=NotificationZMURLOptions
        )

        # validators
        _validate_host_portal = validator("host", "portal", allow_reuse=True, pre=True)(
            validate_no_scheme_url
        )

    class PushoverNotificationSettings(BaseModel):
        class SendAnimations(BaseModel):
            enabled: bool = Field(False)
            token: str = Field(None)
            key: str = Field(None)
        class EndPoints(BaseModel):
            messages: str = Field("/messages.json")
            users: str = Field("/users/validate.json")
            devices: str = Field("/devices.json")
            sounds: str = Field("/sounds.json")
            receipt: str = Field("/receipts/{receipt}.json")
            cancel: str = Field("/cancel/{receipt}.json")
            emergency: str = Field("/emergency.json")

        enabled: bool = Field(False)
        token: str = Field(...)
        key: str = Field(...)
        animation: SendAnimations = Field(default_factory=SendAnimations)
        sounds: Dict[str, str] = Field(default_factory=dict)
        cooldown: float = Field(gt=0.0, default=30.00)
        device: Optional[str] = Field(None)
        url_opts: NotificationZMURLOptions = Field(
            default_factory=NotificationZMURLOptions
        )
        base_url: Optional[AnyUrl] = Field("https://api.pushover.net/1")
        endpoints: EndPoints = Field(default_factory=EndPoints)
        link_url: bool = Field(False)
        link_user: Optional[SecretStr] = Field(None)
        link_pass: Optional[SecretStr] = Field(None)
        priority: int = Field(ge=-2, le=2, default=0)

        # validators
        _validate_host_portal = validator("base_url", allow_reuse=True, pre=True)(
            validate_no_scheme_url
        )

    class ShellScriptNotificationSettings(DefaultNotEnabled):
        script: str = Field(None)

    class WebHookNotificationSettings(DefaultNotEnabled):
        host: AnyUrl = Field(None)
        token: str = Field(None)
        ssl_verify: bool = Field(True)

        # validators
        _validate_host_portal = validator("host", allow_reuse=True, pre=True)(
            validate_no_scheme_url
        )

    class MQTTNotificationSettings(BaseModel):
        class MQTTAnimationSettings(DefaultNotEnabled):
            topic: str = Field('zm_ml/animation')

        class MQTTImageSettings(DefaultNotEnabled):
            topic: str = Field('zm_ml/image')

        enabled: bool = Field(False)
        force: bool = Field(False)
        topic: str = Field("zm_ml/detection")
        broker: Union[IPvAnyAddress, AnyUrl] = Field(None)
        port: int = Field(1883)
        user: str = Field(None)
        pass_: Optional[SecretStr] = Field(None, alias="pass")
        allow_self_signed: bool = Field(False)
        tls_insecure: bool = Field(False)
        tls_ca: Optional[Path] = Field(None)
        tls_cert: Optional[Path] = Field(None)
        tls_key: Optional[Path] = Field(None)
        retain: bool = Field(False)
        qos: int = Field(0)

        image: MQTTImageSettings = Field(default_factory=MQTTImageSettings)
        animation: MQTTAnimationSettings = Field(default_factory=MQTTAnimationSettings)

        # validators
        _validate_host_broker = validator("broker", allow_reuse=True, pre=True)(
            validate_no_scheme_url
        )

    mqtt: MQTTNotificationSettings = Field(default_factory=MQTTNotificationSettings)
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
    webhook: WebHookNotificationSettings = Field(default_factory=WebHookNotificationSettings)


class APIPullMethod(DefaultNotEnabled):
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

        class Debug(DefaultNotEnabled):
            path: Path = Field(Path("/tmp"))

        class Annotations(BaseModel):
            class Zones(DefaultNotEnabled):
                color: Union[str, Tuple[int, int, int]] = Field((255, 0, 0))
                thickness: int = Field(2)

            class Models(DefaultEnabled):
                processor: bool = Field(False)

            zones: Zones = Field(default_factory=Zones)
            model: Models = Field(default_factory=Models)
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
    )(validate_percentage_or_pixels)


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
        validate_percentage_or_pixels
    )


class OverRideStaticObjects(BaseModel):
    enabled: bool = None
    difference: Optional[Union[float, int]] = None
    labels: Optional[List[str]] = Field(None)
    ignore_labels: Optional[List[str]] = Field(None)

    _validate_difference = validator(
        "difference", allow_reuse=True, pre=True, always=True
    )(validate_percentage_or_pixels)


class MatchFilters(BaseModel):
    object: ObjectFilters = Field(default_factory=ObjectFilters)
    face: FaceFilters = Field(default_factory=FaceFilters)
    alpr: AlprFilters = Field(default_factory=AlprFilters)


class OverRideMatchFilters(BaseModel):
    object: OverRideObjectFilters = Field(default_factory=OverRideObjectFilters)
    face: OverRideFaceFilters = Field(default_factory=OverRideFaceFilters)
    alpr: OverRideAlprFilters = Field(default_factory=OverRideAlprFilters)


class MatchStrategy(str, Enum):
    # first match wins
    first = "first"
    most = "most"
    most_models = "most_models"
    most_unique = "most_unique"


class MatchingSettings(BaseModel):
    strategy: MatchStrategy = Field(MatchStrategy.first)
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

    from .validators import validate_resolution, validate_points
    __validate_resolution = validator("resolution", pre=True, allow_reuse=True)(validate_resolution)
    __validate_points = validator("points", pre=True, allow_reuse=True)(validate_points)


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
    variable_data_path: Optional[Path] = Field(Path("/var/lib/zm_ml"))
    tmp_path: Optional[Path] = Field(Path(tempfile.gettempdir()) / "zm_ml")
    thread_workers: Optional[int] = Field(4)


class ConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    config_path: Path = Field(Path("/etc/zm/ml"))
    system: SystemSettings = Field(default_factory=SystemSettings)
    zoneminder: ZMAPISettings = Field(default_factory=ZMAPISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    mlapi: ServerRoutes = Field(default_factory=ServerRoutes)
    animation: AnimationSettings = Field(default_factory=AnimationSettings)
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
