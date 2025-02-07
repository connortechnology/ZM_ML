from __future__ import annotations

import pickle
import time
import uuid
from logging import getLogger
from pathlib import Path
from typing import Optional, Union, List

import cv2
import numpy as np
from sklearn import neighbors

try:
    import dlib
except ImportError:
    dlib = None
try:
    import face_recognition
except ImportError:
    face_recognition = None


from zm_ml.Server.Log import SERVER_LOGGER_NAME
from ..file_locks import FileLock
from ...Models.config import (
    FaceRecognitionLibModelDetectionOptions,
    BaseModelConfig,
    FaceRecognitionLibModelConfig,
    FaceRecognitionLibModelTrainingOptions,
)
from ....Shared.Models.Enums import ModelProcessor, FaceRecognitionLibModelTypes
from ....Shared.Models.config import DetectionResults, Result

logger = getLogger(SERVER_LOGGER_NAME)
LP = "Face_Recognition:"


# Class to handle face recognition
class FaceRecognitionLibDetector(FileLock):
    def __init__(
        self, model_config: Union[BaseModelConfig, FaceRecognitionLibModelConfig]
    ):
        if any([face_recognition is None, dlib is None]):
            raise ImportError(
                f"{LP} face_recognition or dlib not imported, please install face_recognition and dlib!"
            )
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        # Model init params
        self.config: Union[
            BaseModelConfig, FaceRecognitionLibModelConfig
        ] = model_config
        self.detection_options: Optional[
            FaceRecognitionLibModelDetectionOptions
        ] = self.config.detection_options
        self.training_options: Optional[
            FaceRecognitionLibModelTrainingOptions
        ] = self.config.training_options
        self.processor: ModelProcessor = self.config.processor
        self.name: str = self.config.name
        self.knn: Optional[neighbors.KNeighborsClassifier] = None
        self.scaled: bool = False
        self.x_factor: float = 1.0
        self.y_factor: float = 1.0
        self.original_image: Optional[np.ndarray] = None

        self.face_locations: list = []
        self.face_encodings: list = []
        self.trained_faces_file: Optional[Path] = None
        self.processor_check()
        # get trained face encodings loaded
        self.load_trained_faces()
        super().__init__()

    def load_trained_faces(self, faces_file: Optional[Path] = None):
        if faces_file and faces_file.is_file():
            self.trained_faces_file = faces_file
        else:
            self.trained_faces_file = Path(
                f"{self.training_options.dir}/trained_faces.dat"
            )
        logger.debug(f"{LP} loading trained faces from: {self.trained_faces_file}")
        # to increase performance, read encodings from file
        if self.trained_faces_file.is_file():
            logger.debug(
                f"{LP} Trained faces file found. If you want to add new images/people, "
                f"remove: '{self.trained_faces_file}' and retrain"
            )
            try:
                with self.trained_faces_file.open("rb") as f:
                    self.knn = pickle.load(f)
            except Exception as e:
                logger.error(
                    f"{LP} error loading KNN model from '{self.trained_faces_file}' -> {e}"
                )
                raise e
        else:
            logger.warning(f"{LP} trained faces file not found! Please train first!")

    def processor_check(self):
        if self.processor == ModelProcessor.GPU:
            try:
                if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
                    logger.debug(
                        f"{LP} dlib was compiled with CUDA support and there is an available GPU "
                        f"to use for processing! (Total GPUs dlib could use: {dlib.cuda.get_num_devices()})"
                    )
                elif dlib.DLIB_USE_CUDA and not dlib.cuda.get_num_devices() >= 1:
                    logger.error(
                        f"{LP} It appears dlib was compiled with CUDA support but there is not an available GPU "
                        f"for dlib to use! Using CPU for dlib detections..."
                    )
                    self.config.processor = self.processor = ModelProcessor.CPU
                elif not dlib.DLIB_USE_CUDA:
                    logger.error(
                        f"{LP} It appears dlib was not compiled with CUDA support! "
                        f"Using CPU for dlib detections..."
                    )
                    self.config.processor = self.processor = ModelProcessor.CPU
            except Exception as e:
                logger.error(
                    f"{LP} Error checking for CUDA support in dlib! Using CPU for dlib detections... -> {e}"
                )
                self.config.processor = self.processor = ModelProcessor.CPU

    async def detect(self, input_image: np.ndarray):
        result, labels, b_boxes = [], [], []

        if face_recognition is None:
            logger.warning(
                f"{LP} face_recognition library not installed, skipping face detection"
            )
        else:
            max_size: int = self.detection_options.max_size
            detect_start_timer = time.perf_counter()
            h, w = input_image.shape[:2]
            resized_w, resized_h = None, None

            if w > max_size:
                self.scaled = True
                logger.debug(f"{LP} scaling image down using {max_size} as width")
                self.original_image = input_image.copy()
                from ...utils import resize_cv2_image

                input_image = resize_cv2_image(input_image, max_size)
                resized_h, resized_w = input_image.shape[:2]
                self.x_factor = w / resized_w
                self.y_factor = h / resized_h
            _input = f" - model input {resized_w}*{resized_h}" if resized_w else ""
            logger.debug(
                f"{LP}detect: '{self.name}' ({self.processor}) - "
                f"input image {w}*{h}{_input}"
            )
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the target image
            logger.debug(f"{LP}detect: finding faces in image")
            self.acquire_lock()
            self.face_locations = face_recognition.face_locations(
                rgb_image,
                model=self.detection_options.model,
                number_of_times_to_upsample=self.detection_options.upsample_times,
            )
            logger.debug(f"{LP}detect: found {len(self.face_locations)} faces")
            if not len(self.face_locations):
                logger.debug(
                    f"perf:{LP}{self.processor}: computing face locations took "
                    f"{time.perf_counter() - detect_start_timer:.5f} s"
                )
            else:
                logger.debug(f"{LP}detect: finding face encodings")
                self.face_encodings = face_recognition.face_encodings(
                    rgb_image,
                    known_face_locations=self.face_locations,
                    num_jitters=self.detection_options.num_jitters,
                    model="small",  # small is default but faster, large is more accurate
                )

                logger.debug(
                    f"perf:{LP}{self.processor}: computing locations and encodings took "
                    f"{time.perf_counter() - detect_start_timer:.5f} s"
                )

                # TODO: We use KNN for color detection, but we could use it for face recognition too
                ## Create a global KNN object
                if not self.knn:
                    logger.debug(f"{LP} no trained faces found, skipping recognition")
                    if self.scaled:
                        logger.debug(
                            f"{LP} scaling bounding boxes as image was resized for detection"
                        )
                        input_image = self.original_image
                        self.face_locations = self.scale_by_factor(
                            self.face_locations, self.x_factor, self.y_factor
                        )
                    for loc in self.face_locations:
                        label = self.config.unknown_faces.label_as
                        if self.config.unknown_faces.enabled:
                            self.save_unknown_face(loc, input_image)
                        b_boxes.append([loc[3], loc[0], loc[1], loc[2]])
                        labels.append(label)

                else:
                    logger.debug(
                        f"{LP} comparing detected faces to trained (known) faces..."
                    )
                    comparing_timer = time.perf_counter()
                    closest_distances = self.knn.kneighbors(
                        self.face_encodings, n_neighbors=1
                    )
                    logger.debug(
                        f"{LP} closest KNN match indexes (smaller is better): {closest_distances}",
                    )
                    are_matches = [
                        closest_distances[0][i][0]
                        <= self.detection_options.recognition_threshold
                        for i in range(len(self.face_locations))
                    ]
                    prediction_labels = self.knn.predict(self.face_encodings)
                    logger.debug(
                        f"{LP} KNN predictions: {prediction_labels} - are_matches: {are_matches}",
                    )
                    logger.debug(
                        f"perf:{LP}{self.processor}: matching detected faces to known faces took "
                        f"{time.perf_counter() - comparing_timer:.5f} s"
                    )
                    if self.scaled:
                        logger.debug(
                            f"{LP} scaling bounding boxes as image was resized for detection"
                        )
                        input_image = self.original_image
                        self.face_locations = self.scale_by_factor(
                            self.face_locations, self.x_factor, self.y_factor
                        )
                    for pred, loc, rec in zip(
                        prediction_labels, self.face_locations, are_matches
                    ):
                        label = pred if rec else self.config.unknown_faces.label_as
                        if not rec and self.config.unknown_faces.enabled:
                            self.save_unknown_face(loc, input_image)
                        b_boxes.append([loc[3], loc[0], loc[1], loc[2]])
                        labels.append(label)
                    logger.debug(
                        f"perf:{LP} recognition sequence took {time.perf_counter() - comparing_timer:.5f} s"
                    )

            self.release_lock()

            result = DetectionResults(
                success=True if labels else False,
                type=self.config.type_of,
                processor=self.processor,
                name=self.name,
                results=[
                    Result(label=labels[i], confidence=1.0, bounding_box=b_boxes[i])
                    for i in range(len(labels))
                ],
            )

        return result

    @staticmethod
    def scale_by_factor(locations: list, x_factor: float, y_factor: float):
        scaled_face_locations = []
        logger.debug(
            f"scaling bounding boxes by x_factor: {x_factor} and y_factor: {y_factor}"
        )
        logger.debug(f"original bounding boxes: {locations}")
        for loc in locations:
            a, b, c, d = loc
            a = round(a * y_factor)
            b = round(b * x_factor)
            c = round(c * y_factor)
            d = round(d * x_factor)
            scaled_face_locations.append((a, b, c, d))
        logger.debug(f"scaled bounding boxes: {scaled_face_locations}")
        return scaled_face_locations

    def train(self, face_resize_width: Optional[int] = None):
        t = time.perf_counter()
        train_model = self.config.training_options.model
        knn_algo = "ball_tree"
        upsample_times = self.training_options.upsample_times
        num_jitters = self.training_options.num_jitters

        ext = (".jpg", ".jpeg", ".png")
        known_face_encodings = []
        known_face_names = []
        try:
            known_parent_dir = Path(self.config.training_options.dir)
            for train_person_dir in known_parent_dir.glob("*"):
                if train_person_dir.is_dir():
                    for file in train_person_dir.glob("*"):
                        if file.suffix.casefold() in ext:
                            logger.info(
                                f"{LP} training in dir '{train_person_dir}' on '{file}'"
                            )
                            known_face_image = cv2.imread(
                                file.expanduser().resolve().as_posix()
                            )
                            # known_face_image = face_recognition.load_image_file(file)
                            if known_face_image is None or known_face_image.size == 0:
                                logger.error(f"{LP} Error reading file, skipping")
                                continue
                            if not face_resize_width:
                                face_resize_width = self.training_options.max_size
                            logger.debug(f"{LP} resizing to {face_resize_width}")
                            from ...utils import resize_cv2_image

                            known_face_image = resize_cv2_image(
                                known_face_image, face_resize_width
                            )
                            known_face_image = cv2.cvtColor(
                                known_face_image, cv2.COLOR_BGR2RGB
                            )
                            logger.debug(f"{LP} locating faces...")
                            face_locations = face_recognition.face_locations(
                                known_face_image,
                                model=train_model,
                                number_of_times_to_upsample=upsample_times,
                            )
                            if len(face_locations) != 1:
                                extra_err_msg: str = ""
                                if (
                                    self.config.training_options.model
                                    == FaceRecognitionLibModelTypes.HOG
                                ):
                                    extra_err_msg = (
                                        "If you think you have only 1 face try using 'cnn' "
                                        "for training mode. "
                                    )
                                logger.error(
                                    f"{LP} Image has {len(face_locations)} faces, cannot use for training. "
                                    f"We need exactly 1 face. {extra_err_msg}Ignoring..."
                                )
                            else:
                                logger.debug(f"{LP} encoding face...")
                                face_encoding = face_recognition.face_encodings(
                                    known_face_image,
                                    known_face_locations=face_locations,
                                    num_jitters=num_jitters,
                                )
                                if face_encoding:
                                    known_face_encodings.append(face_encoding[0])
                                    known_face_names.append(train_person_dir.name)
                                else:
                                    logger.warning(
                                        f"{LP} no face found in {file} - skipping"
                                    )
                        else:
                            logger.warning(
                                f"{LP} image is not in allowed format! skipping {file}"
                            )
                else:
                    logger.warning(
                        f"{LP} {train_person_dir} is not a directory! skipping"
                    )
        except Exception as e:
            logger.error(f"{LP} Error during recognition training: {e}")
            raise e

        if not len(known_face_names):
            logger.warning(
                f"{LP} No faces found at all, skipping saving of face encodings to file..."
            )
        else:
            import math

            n_neighbors = int(round(math.sqrt(len(known_face_names))))
            logger.debug(f"{LP} KNN using algo: {knn_algo} n_neighbors: {n_neighbors}")
            knn = neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
            )
            timer = time.perf_counter()
            logger.debug(f"{LP} training model ...")
            knn.fit(known_face_encodings, known_face_names)
            logger.debug(
                f"perf:{LP} training model took {time.perf_counter() - timer:.5f} s"
            )
            try:
                with open(self.trained_faces_file, "wb") as f:
                    pickle.dump(knn, f)
            except Exception as exc:
                logger.error(
                    f"{LP} error pickling face encodings to {self.trained_faces_file} -> {exc}"
                )
            else:
                logger.debug(
                    f"{LP} wrote KNN encodings (known faces data) to '{self.trained_faces_file}'"
                )

        logger.debug(
            f"perf:{LP} Recognition training took: {time.perf_counter() - t:.5f} s"
        )

    def save_unknown_face(self, loc: list, input_image: np.ndarray):
        save_dir = Path(self.config.unknown_faces.dir)
        if save_dir.is_dir():
            time_str = time.strftime("%b%d-%Hh%Mm%Ss-")
            unf = f"{save_dir.as_posix()}/{time_str}{uuid.uuid4()}.jpg"
            h, w = input_image.shape[:2]
            leeway = self.config.unknown_faces.leeway_pixels
            x1 = max(
                loc[3] - leeway,
                0,
            )
            y1 = max(
                loc[0] - leeway,
                0,
            )
            x2 = min(
                loc[1] + leeway,
                w,
            )
            y2 = min(
                loc[2] + leeway,
                h,
            )
            crop_img = input_image[y1:y2, x1:x2]
            logger.info(
                f"{LP} saving cropped UNKNOWN '{self.config.unknown_faces.label_as}' face "
                f"at [{x1},{y1},{x2},{y2} - includes leeway of {leeway}px] to {unf}"
            )
            # cv2.imwrite won't throw an exception it outputs a C language WARN to console
            cv2.imwrite(unf, crop_img)
