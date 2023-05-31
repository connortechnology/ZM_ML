import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Pattern, Union, Any, Optional

from pydantic import (
    BaseModel,
    Field,
    AnyUrl,
    validator,
    IPvAnyAddress,
    SecretStr,
    BaseSettings,
)

from .validators import validate_percentage_or_pixels
from ...Shared.Models.validators import (
    validate_no_scheme_url,
    _validate_replace_localhost,
    _validate_file,
    _validate_dir,
)
from ...Shared.Models.config import (
    Testing,
    DefaultEnabled,
    DefaultNotEnabled,
    LoggingSettings,
)
from ..Log import CLIENT_LOGGER_NAME
from ..Models.DEFAULTS import *

logger = logging.getLogger(CLIENT_LOGGER_NAME)


class SystemSettings(BaseModel):
    image_dir: Optional[Path] = Field(Path(DEF_CLNT_SYS_IMAGEDIR))
    config_path: Optional[Path] = Field(Path(DEF_CLNT_SYS_CONFDIR))
    variable_data_path: Optional[Path] = Field(DEF_CLNT_SYS_DATADIR)
    tmp_path: Optional[Path] = Field(Path(DEF_CLNT_SYS_TMPDIR))
    thread_workers: Optional[int] = Field(DEF_CLNT_SYS_THREAD_WORKERS)


class ZoneMinderSettings(BaseSettings):
    class ZMMisc(BaseSettings):
        write_notes: bool = Field(True, env="ML_CLIENT_ZONEMINDER_MISC_WRITE_NOTES")

        class Config:
            extra = "allow"

    misc: ZMMisc = Field(default_factory=ZMMisc)
    portal: Optional[AnyUrl] = Field(None, env="ML_CLIENT_ZONEMINDER_PORTAL")
    api: Optional[AnyUrl] = Field(None, env="ML_CLIENT_ZONEMINDER_API")
    user: Optional[SecretStr] = Field(None, env="ML_CLIENT_ZONEMINDER_USER")
    password: Optional[SecretStr] = Field(None, env="ML_CLIENT_ZONEMINDER_PASSWORD")
    ssl_verify: bool = Field(True, env="ML_CLIENT_ZONEMINDER_SSL_VERIFY")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)

    # validators
    _validate_api_portal = validator("api", "portal", allow_reuse=True, pre=True)(
        validate_no_scheme_url
    )

    class Config:
        extra = "allow"


class ServerRoute(BaseModel):
    name: str = Field(...)
    enabled: bool = Field(True)
    weight: int = Field(0)
    host: AnyUrl = Field(...)
    port: int = Field(5000)
    username: str = None
    password: SecretStr = None
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
    attempt_delay: float = Field(
        ge=0.1, default=2.9, description="Delay between attempts in seconds"
    )


class NotificationZMURLOptions(BaseModel):
    mode: str = Field("jpeg")
    scale: int = Field(50)
    max_fps: int = Field(15)
    buffer: int = Field(1000)
    replay: str = Field("single")


class CoolDownSettings(DefaultNotEnabled):
    seconds: float = Field(
        60.00, ge=0.0, description="Seconds to wait before sending another notification"
    )


class OverRideCoolDownSettings(CoolDownSettings):
    linked: Optional[list[str]] = Field(
        default_factory=list, description="List of linked monitors"
    )


class MLNotificationSettings(BaseModel):
    class ZMNinjaNotificationSettings(BaseModel):
        class ZMNinjaFCMSettings(BaseModel):
            class FCMV1Settings(BaseModel):
                enabled: bool = Field(True)
                key: Optional[SecretStr] = None
                url: Optional[AnyUrl] = None

            v1: FCMV1Settings = Field(default_factory=FCMV1Settings)
            token_file: Path = None
            replace_messages: bool = Field(False)
            date_fmt: str = Field("%I:%M %p, %d-%b")
            android_priority: str = Field("high")
            log_raw_message: bool = Field(False)
            log_message_id: str = None
            android_ttl: int = Field(0)

        enabled: bool = Field(True)
        cooldown: CoolDownSettings = Field(default_factory=CoolDownSettings)
        fcm: ZMNinjaFCMSettings = Field(default_factory=ZMNinjaFCMSettings)

    class GotifyNotificationSettings(BaseModel):
        test_image: bool = Field(False)
        enabled: bool = Field(False)
        host: AnyUrl = None
        token: Optional[str] = None
        portal: AnyUrl = None
        link_url: bool = Field(False)
        link_user: Optional[SecretStr] = None
        link_pass: Optional[SecretStr] = None
        _push_auth: Optional[SecretStr] = None
        cooldown: CoolDownSettings = Field(default_factory=CoolDownSettings)

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
            token: Optional[str] = None
            key: Optional[str] = None

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
        cooldown: CoolDownSettings = Field(default_factory=CoolDownSettings)
        device: Optional[str] = None
        url_opts: NotificationZMURLOptions = Field(
            default_factory=NotificationZMURLOptions
        )
        base_url: Optional[AnyUrl] = Field("https://api.pushover.net/1")
        endpoints: EndPoints = Field(default_factory=EndPoints)
        link_url: bool = Field(False)
        link_user: Optional[SecretStr] = None
        link_pass: Optional[SecretStr] = None
        priority: int = Field(ge=-2, le=2, default=0)

        # validators
        _validate_host_portal = validator("base_url", allow_reuse=True, pre=True)(
            validate_no_scheme_url
        )

    class ShellScriptNotificationSettings(DefaultNotEnabled):
        script: str = None
        cooldown: CoolDownSettings = Field(default_factory=CoolDownSettings)
        I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS: str = "No I am not"

    class MQTTNotificationSettings(BaseModel):
        class MQTTAnimationSettings(DefaultNotEnabled):
            topic: str = Field("zm_ml/animation")

        class MQTTImageSettings(DefaultNotEnabled):
            topic: str = Field("zm_ml/image")

        enabled: bool = Field(False)
        force: bool = Field(False)
        topic: str = Field("zm_ml/detection")
        broker: Optional[Union[IPvAnyAddress, AnyUrl]] = None
        port: int = Field(1883)
        user: str = None
        pass_: Optional[SecretStr] = Field(None, alias="pass")
        allow_self_signed: bool = Field(False)
        tls_insecure: bool = Field(False)
        tls_ca: Optional[Path] = None
        tls_cert: Optional[Path] = None
        tls_key: Optional[Path] = None
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
            path: Optional[Path] = Field(Path("/tmp"))

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


class ObjectFilters(BaseObjectFilters):
    pattern: Optional[Pattern] = None
    labels: Optional[Dict[str, Union[BaseObjectFilters, OverRideObjectFilters]]] = None


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
    labels: Optional[List[str]] = None
    ignore_labels: Optional[List[str]] = None

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

    __validate_resolution = validator("resolution", pre=True, allow_reuse=True)(
        validate_resolution
    )
    __validate_points = validator("points", pre=True, allow_reuse=True)(validate_points)


class MonitorsSettings(BaseModel):
    models: Optional[Dict[str, Any]] = Field(default_factory=dict)
    object_confirm: Optional[bool] = None
    static_objects: Optional[OverRideStaticObjects] = Field(
        default_factory=OverRideStaticObjects
    )
    filters: Optional[OverRideMatchFilters] = Field(
        default_factory=OverRideMatchFilters
    )
    zones: Optional[Dict[str, MonitorZones]] = Field(default_factory=dict)


class ZMDBSettings(BaseSettings):
    host: Union[IPvAnyAddress, AnyUrl, None] = Field(None, env="ML_CLIENT_DB_HOST")
    port: Optional[int] = Field(None, env="ML_CLIENT_DB_PORT")
    user: Optional[str] = Field(None, env="ML_CLIENT_DB_USER")
    password: Optional[SecretStr] = Field(None, env="PASSWORD")
    name: Optional[str] = Field(None, env="ML_CLIENT_DB_NAME")
    driver: Optional[str] = Field(None, env="ML_CLIENT_DB_DRIVER")

    _validate_host = validator("host", allow_reuse=True, pre=True)(
        _validate_replace_localhost
    )

    class Config:
        extra = "allow"


class ConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    config_path: Path = Field(Path("/etc/zm"))
    system: SystemSettings = Field(default_factory=SystemSettings)
    zoneminder: ZoneMinderSettings = Field(default_factory=ZoneMinderSettings)
    db: ZMDBSettings = Field(default_factory=ZMDBSettings)
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

    _validate_config_path = validator(
        "config_path", allow_reuse=True, always=True, pre=True
    )(_validate_dir)


class ClientEnvVars(BaseSettings):
    zm_conf_dir: Path = Field(
        Path("/etc/zm"),
        description="Path to ZoneMinder config files",
        env="ZM_CONF_DIR",
    )
    ml_conf_dir: Optional[Path] = Field(
        None,
        description="Path to ZoneMinder ML config file directory (client/server/secrets .yml)",
        env="ML_CONF_DIR",
    )
    client_conf_file: Optional[Path] = Field(
        None, description="Path to ZM-ML CLIENT config file", env="ML_CLIENT_CONF_FILE"
    )

    db: Optional[ZMDBSettings] = Field(default_factory=ZMDBSettings)
    api: Optional[ZoneMinderSettings] = Field(default_factory=ZoneMinderSettings)

    _validate_client_conf_file = validator(
        "client_conf_file", allow_reuse=True, pre=True, always=True, check_fields=False
    )(_validate_file)
    _validate_zm_conf_dir = validator(
        "zm_conf_dir", allow_reuse=True, pre=True, always=True
    )(_validate_dir)
    _validate_ml_conf_dir = validator(
        "ml_conf_dir", allow_reuse=True, pre=True, always=True
    )(_validate_dir)
