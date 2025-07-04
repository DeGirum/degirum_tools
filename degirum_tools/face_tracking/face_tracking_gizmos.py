#
# face_tracking_gizmos.py: face tracking gizmo classes implementation
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements supplemental classes and gizmos for face extraction, alignment, and re-identification (reID).
#

import cv2
import numpy as np
import threading
import copy
from typing import List, Dict, Any, Optional, ClassVar
from dataclasses import dataclass, asdict, field

from ..result_analyzer_base import clone_result
from ..streams import Gizmo, StreamData, VideoSourceGizmo, tag_inference, tag_video
from ..zone_count import ZoneCounter

from .reid_database import ReID_Database


@dataclass
class FaceStatus:
    """
    Class to hold detected face runtime status.
    """

    attributes: Optional[Any]  # face attributes
    db_id: Optional[str] = None  # database ID
    track_id: int = 0  # face track ID
    last_reid_frame: int = -1  # last frame number on which reID was performed
    next_reid_frame: int = -1  # next frame number on which reID should be performed
    confirmed_count: int = 0  # number of times the face was confirmed
    embeddings: list = field(default_factory=list)  # list of embeddings for the face

    unknown_class: ClassVar[str] = "UNKNOWN"  # class name for unknown faces

    def __str__(self):
        return (
            str(self.attributes)
            if self.attributes is not None
            else FaceStatus.unknown_class
        )

    def to_dict(self):
        return asdict(self)


class ObjectMap:
    """Thread-safe map of object IDs to object attributes."""

    def __init__(self):
        """
        Constructor.
        """

        self._lock = threading.Lock()
        self.map: Dict[int, Any] = {}
        self.alert = False  # flag to indicate if an alert was triggered

    def set_alert(self, alert: bool = True) -> None:
        """
        Set the alert flag.

        Args:
            alert (bool): True to set the alert, False to reset it.
        """
        with self._lock:
            self.alert = alert

    def read_alert(self) -> bool:
        """
        Read the alert flag and reset it.

        Returns:
            bool: True if an alert was triggered, False otherwise.
        """
        with self._lock:
            alert = self.alert
            self.alert = False
            return alert

    def put(self, id: int, value: Any) -> None:
        """
        Add/update an object in the map

        Args:
            id (int): Object ID
            value (Any): Object attributes reference
        """
        with self._lock:
            self.map[id] = value

    def get(self, id: int) -> Optional[Any]:
        """
        Get the object by ID

        Args:
            id (int): The ID of the tracked face.

        Returns:
            Optional[Any]: The deep copy of object attributes or None if not found.
        """
        with self._lock:
            return copy.deepcopy(self.map.get(id))

    def delete(self, expr):
        """
        Delete objects from the map

        Args:
            expr (lambda): logical expression to filter objects to delete
        """
        with self._lock:
            keys_to_delete = [key for key, value in self.map.items() if expr(value)]
            for key in keys_to_delete:
                del self.map[key]


tag_obj_annotate = "object_annotate"  # tag for object annotation meta
tag_face_align = "face_align"  # tag for face alignment and cropping meta
tag_face_search = "face_search"  # tag for face search meta


class ObjectAnnotateGizmo(Gizmo):
    """Object annotating gizmo"""

    lbl_not_tracked = "not tracked"
    lbl_identifying = "identifying"
    lbl_confirming = "confirming"

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_obj_annotate, tag_inference]

    def __init__(
        self,
        object_map: ObjectMap,
        *,
        credence_count: int = 1,
        label_map: dict = {},
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Constructor.

        Args:
            object_map (ObjectMap): The map of object IDs to attributes.
            credence_count (int): Number of times the face is recognized before confirming it.
            label_map (dict): Map of special labels to their display names. Recognized keys: "not tracked", "identifying", "confirming".
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._object_map = object_map
        self._credence_count = credence_count
        self._label_map = label_map
        self._label_map.setdefault(
            ObjectAnnotateGizmo.lbl_not_tracked, ObjectAnnotateGizmo.lbl_not_tracked
        )
        self._label_map.setdefault(
            ObjectAnnotateGizmo.lbl_identifying, ObjectAnnotateGizmo.lbl_identifying
        )
        self._label_map.setdefault(
            ObjectAnnotateGizmo.lbl_confirming, ObjectAnnotateGizmo.lbl_confirming
        )

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have face detection gizmo in upstream"
                )

            clone = clone_result(result)

            for r in clone.results:
                track_id = r.get("track_id")
                if track_id is None:
                    r["label"] = self._label_map[ObjectAnnotateGizmo.lbl_not_tracked]
                else:
                    obj_status = self._object_map.get(track_id)
                    if obj_status is None:
                        r["label"] = self._label_map[
                            ObjectAnnotateGizmo.lbl_identifying
                        ]
                    else:
                        if obj_status.confirmed_count < self._credence_count:
                            r["label"] = self._label_map[
                                ObjectAnnotateGizmo.lbl_confirming
                            ]
                        elif obj_status.attributes is not None:
                            r["attributes"] = obj_status.attributes
                            r["label"] = str(obj_status)
                        else:
                            # unknown face
                            r["label"] = FaceStatus.unknown_class
                            self._object_map.set_alert()

            new_meta = data.meta.clone()
            new_meta.append(clone, self.get_tags())
            self.send_result(StreamData(data, new_meta))


class FaceExtractGizmo(Gizmo):
    """Face extracting and aligning gizmo"""

    # meta keys
    key_original_result = "original_result"  # original AI object detection result
    key_cropped_result = "cropped_result"  # sub-result for particular crop
    key_cropped_index = "cropped_index"  # the number of that sub-result
    key_is_last_crop = "is_last_crop"  # 'last crop in the frame' flag

    def __init__(
        self,
        target_image_size: int,
        face_reid_map: Optional[ObjectMap] = None,
        reid_expiration_frames: int = 0,
        zone_ids: Optional[List[int]] = None,
        min_face_size: int = 0,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """
        Constructor.

        Args:
            target_image_size (int): Size to which the image should be resized.
            face_reid_map (ObjectMap): The map of face IDs to face attributes; used for filtering. None means no filtering.
            reid_expiration_frames (int): Number of frames after which the face reID needs to be repeated.
            zone_ids (List[int]): List of zone IDs to filter the faces. None means no filtering.
            min_face_size (int): Minimum size of the smaller side of the face bbox in pixels to be considered for reID. 0 means no filtering.
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._image_size = target_image_size
        self._face_reid_map = face_reid_map
        self._reid_expiration_frames = reid_expiration_frames
        self._zone_ids = zone_ids
        self._min_face_size = min_face_size

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_align]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            # get inference result
            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have face detection gizmo in upstream"
                )

            # get current frame ID
            video_meta = data.meta.find_last(tag_video)
            if video_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream"
                )
            frame_id = video_meta[VideoSourceGizmo.key_frame_id]

            for i, r in enumerate(result.results):

                landmarks = r.get("landmarks")
                if not landmarks or len(landmarks) != 5:
                    continue

                # apply filtering based on the face size
                if self._min_face_size > 0:
                    bbox = r.get("bbox")
                    if bbox is not None:
                        w, h = abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])
                        if min(w, h) < self._min_face_size:
                            continue  # skip if the face is too small

                # apply filtering based on zone IDs
                if self._zone_ids:
                    in_zone = r.get(ZoneCounter.key_in_zone)
                    if in_zone is None or all(
                        not in_zone[zid] for zid in self._zone_ids if zid < len(in_zone)
                    ):
                        # skip if the face is not in the specified zones
                        continue

                # get the track ID and skip if not available
                track_id = r.get("track_id")
                if track_id is None:
                    # no track ID - skip reID
                    continue

                # apply filtering based on the face reID map
                if self._face_reid_map is not None:
                    face_status = self._face_reid_map.get(track_id)
                    if face_status is None:
                        # new face
                        face_status = FaceStatus(
                            attributes=None,
                            track_id=track_id,
                            last_reid_frame=frame_id,
                            next_reid_frame=frame_id
                            + self._reid_expiration_frames // 4,
                        )
                    else:
                        if frame_id < face_status.next_reid_frame:
                            # skip reID if the face is already in the map and not expired
                            continue

                        delta = min(
                            self._reid_expiration_frames,
                            2
                            * (
                                face_status.next_reid_frame
                                - face_status.last_reid_frame
                            ),
                        )
                        face_status.last_reid_frame = frame_id
                        face_status.next_reid_frame = frame_id + delta
                    self._face_reid_map.put(track_id, face_status)

                keypoints = [np.array(lm["landmark"]) for lm in landmarks]

                crop_img = FaceExtractGizmo.face_align_and_crop(
                    data.data, keypoints, self._image_size
                )

                crop_obj = copy.deepcopy(r)
                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: crop_obj,
                    self.key_cropped_index: i,
                    self.key_is_last_crop: i == len(result.results) - 1,
                }
                new_meta = data.meta.clone()
                new_meta.remove_last(tag_inference)
                new_meta.append(crop_meta, self.get_tags())
                self.send_result(StreamData(crop_img, new_meta))

            # delete expired faces from the map
            if self._reid_expiration_frames > 0 and self._face_reid_map is not None:
                self._face_reid_map.delete(
                    lambda x: x.last_reid_frame + self._reid_expiration_frames
                    < frame_id
                )

    @staticmethod
    def face_align_and_crop(img: np.ndarray, landmarks: list, image_size) -> np.ndarray:
        """
        Align and crop the face from the image based on the given landmarks.

        Args:
            img (np.ndarray): The full image (not the cropped bounding box).
            landmarks (List[np.ndarray]): List of 5 keypoints (landmarks) as (x, y) coordinates in the following order:
                [left eye, right eye, nose, left mouth, right mouth].
            image_size (int): The size to which the image should be resized.

        Returns:
            np.ndarray: the aligned face image
        """

        # reference keypoints for alignment:
        # these are the coordinates of the 5 keypoints in the reference image (112x112);
        # the order is: left eye, right eye, nose, left mouth, right mouth
        _arcface_ref_kps = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        assert len(landmarks) == 5
        dst = _arcface_ref_kps * image_size / 112.0  # scale to the target size

        M, _ = cv2.estimateAffinePartial2D(np.array(landmarks), dst, method=cv2.LMEDS)

        aligned_img = cv2.warpAffine(img, M, [image_size, image_size])
        return aligned_img


class FaceSearchGizmo(Gizmo):
    """Face reID search gizmo"""

    def __init__(
        self,
        face_reid_map: ObjectMap,
        db: ReID_Database,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
        accumulate_embeddings: bool = False,
    ):
        """
        Constructor.

        Args:
            face_reid_map (ObjectMap): The map of face IDs to face attributes.
            db (ReID_Database): vector database object
            stream_depth (int): Depth of the stream.
            allow_drop (bool): Whether to allow dropping frames.
            accumulate_embeddings (bool): Whether to accumulate embeddings in the face map.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._face_reid_map = face_reid_map
        self._db = db
        self._accumulate_embeddings = accumulate_embeddings

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_face_search]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            # get inference result
            result = data.meta.find_last(tag_inference)
            if (
                result is None
                or not result.results
                or result.results[0].get("data") is None
            ):
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have reID inference gizmo in upstream"
                )

            # get current frame ID
            video_meta = data.meta.find_last(tag_video)
            if video_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream"
                )

            # get face crop result
            crop_meta = data.meta.find_last(tag_face_align)
            if crop_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: crop meta not found: you need to have {FaceExtractGizmo.__class__.__name__} in upstream"
                )

            face_obj = crop_meta.get(FaceExtractGizmo.key_cropped_result)
            assert face_obj
            track_id = face_obj.get("track_id")
            assert track_id

            # search the database for the face embedding
            embedding = result.results[0].get("data").ravel()
            embedding /= np.linalg.norm(embedding)
            db_id, attributes = self._db.get_attributes_by_embedding(embedding)

            # update the face attributes in the map
            face = self._face_reid_map.get(track_id)
            if face is not None:
                # existing face - update the attributes
                if face.db_id == db_id:
                    face.confirmed_count += 1
                else:
                    face.confirmed_count = 0
                face.attributes = attributes
                face.db_id = db_id
                self._face_reid_map.put(track_id, face)

                if self._accumulate_embeddings:
                    face.embeddings.append(embedding)
