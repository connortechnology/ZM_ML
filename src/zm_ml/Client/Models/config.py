import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Pattern, Union, Any, Optional

from pydantic import (
    BaseModel,
    Field,
    AnyUrl,
    field_validator,
    IPvAnyAddress,
    SecretStr,
)
from pydantic_settings import BaseSettings

from .validators import validate_percentage_or_pixels, validate_resolution, validate_points
from ...Shared.Models.validators import (
    validate_no_scheme_url,
    validate_replace_localhost,
    validate_file,
    validate_dir,
    validate_not_enabled,
    validate_enabled
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


class ZMDBSettings(BaseSettings, extra="allow"):
    host: Union[IPvAnyAddress, AnyUrl, None] = Field(None, env="ML_CLIENT_DB_HOST")
    port: Optional[int] = Field(None, env="ML_CLIENT_DB_PORT")
    user: Optional[str] = Field(None, env="ML_CLIENT_DB_USER")
    password: Optional[SecretStr] = Field(None, env="PASSWORD")
    name: Optional[str] = Field(None, env="ML_CLIENT_DB_NAME")
    driver: Optional[str] = Field(None, env="ML_CLIENT_DB_DRIVER")

    _validate_host = field_validator("host", mode="before")(
        validate_replace_localhost
    )


class SystemSettings(BaseModel):
    image_dir: Optional[Path] = Field(Path(DEF_CLNT_SYS_IMAGEDIR))
    config_path: Optional[Path] = Field(Path(DEF_CLNT_SYS_CONFDIR))
    variable_data_path: Optional[Path] = Field(DEF_CLNT_SYS_DATADIR)
    tmp_path: Optional[Path] = Field(Path(DEF_CLNT_SYS_TMPDIR))
    thread_workers: Optional[int] = Field(DEF_CLNT_SYS_THREAD_WORKERS)


class ZoneMinderSettings(BaseSettings, extra="allow"):
    class ZMMisc(BaseSettings):
        write_notes: bool = Field(True, env="ML_CLIENT_ZONEMINDER_MISC_WRITE_NOTES")


    class ZMAPISettings(BaseSettings):
        api_url: Optional[AnyUrl] = Field(None, env="ML_CLIENT_ZONEMINDER_API_URL")
        user: Optional[SecretStr] = Field(None, env="ML_CLIENT_ZONEMINDER_API_USER")
        password: Optional[SecretStr] = Field(None, env="ML_CLIENT_ZONEMINDER_API_PASSWORD")
        ssl_verify: Optional[bool] = Field(True, env="ML_CLIENT_ZONEMINDER_API_SSL_VERIFY")
        headers: Optional[Dict] = Field(default_factory=dict, env="ML_CLIENT_ZONEMINDER_API_HEADERS")

        _validate_api_url = field_validator("api_url", mode="before")(
            validate_no_scheme_url
        )

    portal_url: Optional[AnyUrl] = Field(None, env="ML_CLIENT_ZONEMINDER_PORTAL_URL")
    misc: Optional[ZMMisc] = Field(default_factory=ZMMisc)
    api: Optional[ZMAPISettings] = Field(default_factory=ZMAPISettings)
    db: Optional[ZMDBSettings] = Field(default_factory=ZMDBSettings)

    _validate_portal_url = field_validator("portal_url", mode="before")(
        validate_no_scheme_url
    )



class ServerRoute(DefaultEnabled):
    # Make 1 attr required so empty entries will fail.
    name: str = Field(...)
    weight: Optional[int] = Field(0)
    host: AnyUrl = Field(...)
    port: Optional[int] = Field(5000)
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    timeout: Optional[int] = Field(90, ge=0)

    # validators
    _validate_mlapi_host_localhost = field_validator("host", mode="before")(
        validate_replace_localhost
    )
    _validate_mlapi_host_no_scheme = field_validator("host", mode="before")(
        validate_no_scheme_url
    )


class ServerRoutes(BaseModel):
    routes: List[ServerRoute] = Field(default_factory=list)


class AnimationSettings(BaseModel):
    gif: bool = Field(False)
    mp4: bool = Field(False)
    width: Optional[int] = Field(640)
    fast_gif: bool = Field(False)
    low_memory: Optional[bool] = Field(False)
    overwrite: Optional[bool] = Field(False)
    max_attempts: Optional[int] = Field(ge=1, default=3)
    attempt_delay: Optional[float] = Field(
        ge=0.1, default=2.9, description="Delay between attempts in seconds"
    )


class NotificationZMURLOptions(BaseModel):
    mode: Optional[str] = Field("jpeg")
    scale: Optional[int] = Field(50)
    max_fps: Optional[int] = Field(15)
    buffer: Optional[int] = Field(1000)
    replay: Optional[str] = Field("single")


class CoolDownSettings(DefaultNotEnabled):
    seconds: Optional[float] = Field(
        60.00, ge=0.0, description="Seconds to wait before sending another notification"
    )


class OverRideCoolDownSettings(CoolDownSettings):
    linked: Optional[list[str]] = Field(
        default_factory=list, description="List of linked monitors"
    )


class MLNotificationSettings(BaseModel):
    class ZMNinjaNotificationSettings(DefaultEnabled):
        class ZMNinjaFCMSettings(BaseModel):
            class FCMV1Settings(DefaultEnabled):
                key: Optional[SecretStr] = None
                url: Optional[AnyUrl] = None

            v1: Optional[FCMV1Settings] = Field(default_factory=FCMV1Settings)
            token_file: Optional[Path] = None
            replace_messages: Optional[bool] = Field(False)
            date_fmt: Optional[str] = Field("%I:%M %p, %d-%b")
            android_priority: Optional[str] = Field("high")
            log_raw_message: Optional[bool] = Field(False)
            log_message_id: Optional[str] = None
            android_ttl: Optional[int] = Field(0)

        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)
        fcm: Optional[ZMNinjaFCMSettings] = Field(default_factory=ZMNinjaFCMSettings)

    class GotifyNotificationSettings(DefaultNotEnabled):
        test_image: Optional[bool] = Field(False)
        host: Optional[AnyUrl] = None
        token: Optional[str] = None
        portal: Optional[AnyUrl] = None
        clickable_link: Optional[bool] = Field(False)
        link_user: Optional[SecretStr] = None
        link_pass: Optional[SecretStr] = None
        _push_auth: Optional[SecretStr] = None
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)

        url_opts: Optional[NotificationZMURLOptions] = Field(
            default_factory=NotificationZMURLOptions
        )

        # validators
        _validate_host_portal = field_validator("host", "portal", mode="before")(
            validate_no_scheme_url
        )

    class PushoverNotificationSettings(BaseModel):
        class SendAnimations(DefaultNotEnabled):
            token: Optional[str] = None
            key: Optional[str] = None

        class EndPoints(BaseModel):
            messages: Optional[str] = Field("/messages.json")
            users: Optional[str] = Field("/users/validate.json")
            devices: Optional[str] = Field("/devices.json")
            sounds: Optional[str] = Field("/sounds.json")
            receipt: Optional[str] = Field("/receipts/{receipt}.json")
            cancel: Optional[str] = Field("/cancel/{receipt}.json")
            emergency: Optional[str] = Field("/emergency.json")

        enabled: Optional[bool] = Field(False)
        token: str = Field(...)
        key: str = Field(...)
        animation: Optional[SendAnimations] = Field(default_factory=SendAnimations)
        sounds: Optional[Dict[str, str]] = Field(default_factory=dict)
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)
        device: Optional[str] = None
        url_opts: Optional[NotificationZMURLOptions] = Field(
            default_factory=NotificationZMURLOptions
        )
        base_url: Optional[AnyUrl] = Field("https://api.pushover.net/1")
        endpoints: Optional[EndPoints] = Field(default_factory=EndPoints)
        clickable_link: Optional[bool] = Field(False)
        link_user: Optional[SecretStr] = None
        link_pass: Optional[SecretStr] = None
        priority: Optional[int] = Field(ge=-2, le=2, default=0)

        # validators
        _validate_host_portal = field_validator("base_url", mode="before")(
            validate_no_scheme_url
        )

    class ShellScriptNotificationSettings(DefaultNotEnabled):
        script: Optional[str] = None
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)
        I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS: Optional[str] = "No I am not"
        args: Optional[List[str]] = None

    class MQTTNotificationSettings(DefaultNotEnabled):

        class MQTTImageSettings(DefaultNotEnabled):
            format: Optional[str] = Field("base64", regex="^(bytes|base64)$")
            retain: Optional[bool] = True

        keep_alive: Optional[int] = Field(60, ge=1)
        root_topic: Optional[str] = Field("zm_ml")
        broker: Optional[str] = None
        port: Optional[int] = Field(1883)
        user: Optional[str] = None
        pass_: Optional[SecretStr] = Field(None, alias="pass")
        tls_secure: Optional[bool] = Field(True)
        tls_ca: Optional[Path] = None
        tls_cert: Optional[Path] = None
        tls_key: Optional[Path] = None
        retain: Optional[bool] = Field(False)
        qos: Optional[int] = Field(0)
        cooldown: Optional[CoolDownSettings] = Field(default_factory=CoolDownSettings)

        image: Optional[MQTTImageSettings] = Field(default_factory=MQTTImageSettings)

    mqtt: Optional[MQTTNotificationSettings] = Field(default_factory=MQTTNotificationSettings)
    zmninja: Optional[ZMNinjaNotificationSettings] = Field(
        default_factory=ZMNinjaNotificationSettings
    )
    gotify: Optional[GotifyNotificationSettings] = Field(
        default_factory=GotifyNotificationSettings
    )
    pushover: Optional[PushoverNotificationSettings] = Field(
        default_factory=PushoverNotificationSettings
    )
    shell_script: Optional[ShellScriptNotificationSettings] = Field(
        default_factory=ShellScriptNotificationSettings
    )


class APIPullMethod(DefaultNotEnabled):
    fps: Optional[int] = Field(1)
    attempts: Optional[int] = Field(3)
    delay: Optional[float] = Field(1.0)
    check_snapshots: Optional[bool] = Field(True)
    snapshot_frame_skip: Optional[int] = Field(3)
    max_frames: Optional[int] = Field(0)


class DetectionSettings(BaseModel):
    class ImageSettings(BaseModel):
        class PullMethod(BaseModel):
            shm: Optional[bool] = Field(False)
            api: Optional[APIPullMethod] = Field(default_factory=APIPullMethod)
            zmu: Optional[bool] = Field(False)
            zms: Optional[bool] = Field(False)

            _validate_ = field_validator("shm", "zmu", "zms", mode="before")(validate_not_enabled)

        class Debug(DefaultNotEnabled):
            path: Optional[Path] = Field(Path("/tmp"))

        class Annotations(BaseModel):
            class Zones(DefaultNotEnabled):
                color: Union[str, Tuple[int, int, int], None] = Field((255, 0, 0))
                thickness: Optional[int] = Field(2)

            class Models(DefaultEnabled):
                processor: Optional[bool] = Field(False)

            zones: Optional[Zones] = Field(default_factory=Zones)
            model: Optional[Models] = Field(default_factory=Models)
            confidence: Optional[bool] = Field(True)

        class Training(DefaultEnabled):
            from tempfile import gettempdir

            enabled: Optional[bool] = Field(False)
            path: Optional[Path] = Field(Path(gettempdir()) / "src/training")

        pull_method: Optional[PullMethod] = Field(default_factory=PullMethod)
        debug: Optional[Debug] = Field(default_factory=Debug)
        annotation: Optional[Annotations] = Field(default_factory=Annotations)
        training: Optional[Training] = Field(default_factory=Training)

    models: Optional[Dict] = Field(default_factory=dict)
    import_zones: Optional[bool] = Field(False)
    match_origin_zone: Optional[bool] = Field(False)
    images: Optional[ImageSettings] = Field(default_factory=ImageSettings)


class BaseObjectFilters(BaseModel):
    min_conf: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    total_max_area: Union[float, int, str, None] = Field(default=None)
    total_min_area: Union[float, int, str, None] = Field(default=None)
    max_area: Union[float, int, str, None] = Field(default=None)
    min_area: Union[float, int, str, None] = Field(default=None)

    # validators
    _normalize_areas = field_validator(
        "total_max_area", "total_min_area", "max_area", "min_area"
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
    labels: Optional[List[str]] = Field(default_factory=list)
    ignore_labels: Optional[List[str]] = Field(default_factory=list)

    _validate_difference = field_validator("difference")(
        validate_percentage_or_pixels
    )


class OverRideStaticObjects(BaseModel):
    enabled: Optional[bool] = None
    difference: Optional[Union[float, int]] = None
    labels: Optional[List[str]] = None
    ignore_labels: Optional[List[str]] = None

    _validate_difference = field_validator(
        "difference", mode="before"
    )(validate_percentage_or_pixels)


class MatchFilters(BaseModel):
    object: Optional[ObjectFilters] = Field(default_factory=ObjectFilters)
    face: Optional[FaceFilters] = Field(default_factory=FaceFilters)
    alpr: Optional[AlprFilters] = Field(default_factory=AlprFilters)


class OverRideMatchFilters(BaseModel):
    object: Optional[OverRideObjectFilters] = Field(default_factory=OverRideObjectFilters)
    face: Optional[OverRideFaceFilters] = Field(default_factory=OverRideFaceFilters)
    alpr: Optional[OverRideAlprFilters] = Field(default_factory=OverRideAlprFilters)


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
    points: Optional[List[Tuple[int, int]]] = None
    resolution: Optional[Tuple[int, int]] = None
    object_confirm: Optional[bool] = None
    static_objects: Union[OverRideStaticObjects, StaticObjects, None] = Field(
        default_factory=OverRideStaticObjects
    )
    filters: Union[MatchFilters, OverRideMatchFilters, None] = Field(
        default_factory=OverRideMatchFilters
    )
    __validate_resolution = field_validator("resolution", mode="before")(
        validate_resolution
    )
    __validate_points = field_validator("points", mode="before")(validate_points)


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


class ConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    config_path: Path = Field(Path("/etc/zm"))
    system: SystemSettings = Field(default_factory=SystemSettings)
    zoneminder: ZoneMinderSettings = Field(default_factory=ZoneMinderSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    mlapi: ServerRoutes = Field(default_factory=ServerRoutes)
    animation: AnimationSettings = Field(default_factory=AnimationSettings)
    notifications: MLNotificationSettings = Field(
        default_factory=MLNotificationSettings
    )
    label_groups: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    detection_settings: DetectionSettings = Field(default_factory=DetectionSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)
    monitors: Optional[Dict[int, MonitorsSettings]] = Field(default_factory=dict)

    _validate_config_path = field_validator(
        "config_path", mode="before"
    )(validate_dir)


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

    _validate_client_conf_file = field_validator(
        "client_conf_file", mode="before", check_fields=False
    )(validate_file)
    _validate_zm_conf_dir = field_validator(
        "zm_conf_dir", "ml_conf_dir", mode="before"
    )(validate_dir)

