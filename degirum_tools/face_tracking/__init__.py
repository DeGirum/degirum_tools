#
# Face tracking application package
# Copyright DeGirum Corp. 2025
#
# Implements various classes and functions for face tracking application development
#

import os
import tempfile
import degirum as dg
from typing import Optional, Tuple, Union

from ..environment import get_token
from ..event_detector import EventDetector
from ..inference_support import attach_analyzers
from ..math_support import AnchorPoint, compute_kmeans
from ..notifier import EventNotifier, notification_config_console
from ..object_tracker import ObjectTracker
from ..object_storage_support import ObjectStorage, ObjectStorageConfig
from ..streams import (
    Composition,
    Watchdog,
    AiAnalyzerGizmo,
    AiSimpleGizmo,
    VideoDisplayGizmo,
    VideoSaverGizmo,
    VideoSourceGizmo,
    VideoStreamerGizmo,
)
from ..zone_count import ZoneCounter

from .face_tracking_gizmos import (
    ObjectMap,
    FaceSearchGizmo,
    FaceExtractGizmo,
    ObjectAnnotateGizmo,
)
from .reid_database import ReID_Database


class FaceTracking:

    annotated_video_suffix = "_annotated"  # suffix for annotated video clips

    def __init__(
        self,
        *,
        hw_location: str,
        model_zoo_url: str,
        face_detector_model_name: str,
        face_reid_model_name: str,
        clip_storage_config: ObjectStorageConfig,
        db_filename: str,
        token: Optional[str] = None,
        face_detector_model_devices: Optional[list] = None,
        face_reid_model_devices: Optional[list] = None,
    ):
        """
        Constructor.

        Args:
            hw_location (str): Hardware location for the inference.
            model_zoo_url (str): URL of the model zoo.
            face_detector_model_name (str): Name of the face detection model in the model zoo.
            face_reid_model_name (str): Name of the face reID model in the model zoo.
            clip_storage_config (ObjectStorageConfig): Configuration for the object storage where video clips are stored.
            db_filename (str): Path to the reID database.
            token (str, optional): cloud API token or None to use the token from environment.
            face_detector_model_devices (Optional[list]): List of device indexes for the face detector model. If None, all devices are used.
            face_reid_model_devices (Optional[list]): List of device indexes for the face reID model. If None, all devices are used.
        """
        self._hw_location = hw_location
        self._model_zoo_url = model_zoo_url
        self._face_detector_model_name = face_detector_model_name
        self._face_detector_model_devices = face_detector_model_devices
        self._face_reid_model_name = face_reid_model_name
        self._face_reid_model_devices = face_reid_model_devices
        self._clip_storage_config = clip_storage_config
        self._db_filename = db_filename
        self._token = token
        self.db: Optional[ReID_Database] = None
        self._open_db()

    def _open_db(self):
        """
        Open the database for face reID.
        If the database does not exist, create it.
        """
        if self.db is None:
            self.db = ReID_Database(self._db_filename)
        return self.db

    def _load_models(self, zone, reid_expiration_frames):
        """
        Load the face detection and face reID models from the model zoo.
        """
        zoo = dg.connect(
            self._hw_location,
            self._model_zoo_url,
            get_token() if self._token is None else self._token,
        )
        face_detect_model = zoo.load_model(self._face_detector_model_name)
        if self._face_detector_model_devices:
            face_detect_model.devices_selected = self._face_detector_model_devices

        face_reid_model = zoo.load_model(
            self._face_reid_model_name, non_blocking_batch_predict=True
        )
        if self._face_reid_model_devices:
            face_reid_model.devices_selected = self._face_reid_model_devices

        # face tracker
        object_tracker = ObjectTracker(
            track_thresh=0.35,
            track_buffer=reid_expiration_frames + 1,
            match_thresh=0.9999,
            trail_depth=reid_expiration_frames + 1,
            anchor_point=AnchorPoint.CENTER,
            show_overlay=True,
            show_only_track_ids=True,
        )

        # in-zone counter for all faces
        all_objects_zone_counter = ZoneCounter(
            [zone],
            triggering_position=AnchorPoint.CENTER,
            show_overlay=True,
        )

        # attach tracker and zone counter analyzers to the face detection model
        attach_analyzers(face_detect_model, [object_tracker, all_objects_zone_counter])

        return face_detect_model, face_reid_model

    def list_clips(self):
        """
        List the video clips in the storage.
        Returns a dictionary where the key is the clip filename and value is the list of
            video clip file objects (of minio.Object type) associated with that clip (original video clip, JSON annotations, annotated video clip)
        """

        ret: dict = {}
        storage = ObjectStorage(self._clip_storage_config)
        for f in storage.list_bucket_contents():
            if f.object_name.endswith(".mp4"):
                if FaceTracking.annotated_video_suffix not in f.object_name:
                    key = f.object_name.replace(".mp4", "")
                    ret.setdefault(key, {})["original"] = f
                else:
                    key = f.object_name.replace(
                        FaceTracking.annotated_video_suffix, ""
                    ).replace(".mp4", "")
                    ret.setdefault(key, {})["annotated"] = f
            elif f.object_name.endswith(".json"):
                key = f.object_name.replace(".json", "")
                ret.setdefault(key, {})["json"] = f
            else:
                continue

        return ret

    def download_clip(self, filename: str) -> bytes:
        """
        Download the video clip from the storage.

        Args:
            filename (str): The name of the video clip to download.

        Returns:
            bytes: The bytes of the downloaded video clip.
        """
        storage = ObjectStorage(self._clip_storage_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, filename)
            storage.download_file_from_object_storage(filename, local_path)
            return open(local_path, "rb").read()

    def remove_clip(self, filename: str):
        """
        Remove the video clip from the storage.

        Args:
            filename (str): The name of the video clip to remove.
        """
        storage = ObjectStorage(self._clip_storage_config)
        storage.delete_file_from_object_storage(filename)

    def run_analysis(self, clip, zone):
        """
        Run the face analysis pipeline on a video clip.

        Args:
            clip (minio.Object): The video clip file object.
            zone (list): Zone coordinates for in-zone counting.

        Returns:
            ObjectMap: The map of face IDs to face objects found in the clip. Each face object includes a table of embeddings.
        """

        with tempfile.TemporaryDirectory() as tmpdir:

            # define the storage and paths
            storage = ObjectStorage(self._clip_storage_config)
            dirname = os.path.dirname(clip.object_name)
            filename = os.path.basename(clip.object_name)
            file_stem, file_ext = os.path.splitext(filename)
            out_filename = file_stem + FaceTracking.annotated_video_suffix + file_ext
            out_object_name = ((dirname + "/") if dirname else "") + out_filename
            input_video_local_path = os.path.join(tmpdir, filename)
            output_video_local_path = os.path.join(tmpdir, out_filename)

            # download the clip to local storage
            storage.download_file_from_object_storage(
                clip.object_name, input_video_local_path
            )

            # load models
            face_detect_model, face_reid_model = self._load_models(zone, 10)
            reid_height = face_reid_model.input_shape[0][1]  # reID model input height

            # suppress all annotations
            face_detect_model.overlay_line_width = 0
            face_detect_model.overlay_show_labels = True
            face_detect_model.overlay_show_probabilities = False

            #
            # define gizmos
            #

            # video source gizmo
            source = VideoSourceGizmo(input_video_local_path, retry_on_error=True)

            # face detector AI gizmo
            face_detect = AiSimpleGizmo(face_detect_model)

            face_map = ObjectMap()  # object map for face attributes

            # face crop gizmo
            face_extract = FaceExtractGizmo(
                target_image_size=reid_height,
                face_reid_map=face_map,
                reid_expiration_frames=0,
                zone_ids=[0],
                min_face_size=reid_height // 2,
            )

            # face ReID AI gizmo
            face_reid = AiSimpleGizmo(face_reid_model)

            # face reID search gizmo
            face_search = FaceSearchGizmo(
                face_map, self._open_db(), credence_count=1, accumulate_embeddings=True
            )

            # object annotator gizmo
            face_annotate = ObjectAnnotateGizmo(face_map)

            # annotated video saved gizmo
            saver = VideoSaverGizmo(output_video_local_path, show_ai_overlay=True)

            #
            # define pipeline and run it
            #
            Composition(
                source >> face_detect >> face_extract >> face_reid >> face_search,
                face_detect >> face_annotate >> saver,
            ).start()

            # upload the annotated video to the object storage
            storage.upload_file_to_object_storage(
                output_video_local_path, out_object_name
            )

            # compute K-means clustering on the embeddings
            for id, face in face_map.map.items():
                face.embeddings = compute_kmeans(face.embeddings)

            return face_map

    def run_tracking(
        self,
        video_source,
        *,
        zone,
        clip_duration=100,
        reid_expiration_frames=10,
        credence_count=4,
        notification_config=notification_config_console,
        notification_message="{time}: Unknown person detected (saved video: {url})",
        local_display=True,
        stream_name="Face Tracking",
    ) -> Tuple[Composition, Watchdog]:
        """
        Run the face tracking pipeline.

        Args:
            video_source (Any): Path to the video file or camera index.
            zone (list): Zone coordinates for in-zone counting.
            clip_duration (int): Duration of the clip in frames for saving clips.
            reid_expiration_frames (int): Number of frames after which the face reID needs to be repeated.
            credence_count (int): Number of frames to consider a face as known.
            notification_config (str): Apprise configuration string for notifications.
            notification_message (str): Message template for notifications.
            local_display (bool): Whether to display the video locally or by RTSP stream.
            stream_name (str): Window title for local display or URL path for RTSP streaming.

        Returns:
            tuple: A tuple containing:
                - Composition: The pipeline composition object.
                - Watchdog: Watchdog object to monitor the pipeline.
        """

        # load models
        face_detect_model, face_reid_model = self._load_models(
            zone, reid_expiration_frames
        )

        face_detect_model.overlay_line_width = 1
        reid_height = face_reid_model.input_shape[0][1]  # reID model input height
        face_map = ObjectMap()  # object map for face attributes

        #
        # create analyzers
        #

        # "unknown face in zone" event detector
        unknown_face_event_name = "unknown_face"  # name of the event to be generated
        unknown_face_event_detector = EventDetector(
            f"""
            Trigger: {unknown_face_event_name}
            when: CustomMetric
            is greater than: 0
            during: [1, frame]
            """,
            custom_metric=lambda result, params: int(face_map.read_alert()),
            show_overlay=False,
        )

        unknown_face_notifier = EventNotifier(
            "Unknown person detected",
            unknown_face_event_name,
            message=notification_message,
            notification_config=notification_config,
            clip_save=True,
            clip_duration=clip_duration,
            clip_pre_trigger_delay=clip_duration // 2,
            clip_embed_ai_annotations=False,
            storage_config=self._clip_storage_config,
        )

        #
        # define gizmos
        #

        # video source gizmo
        source = VideoSourceGizmo(video_source)

        # face detector AI gizmo
        face_detect = AiSimpleGizmo(face_detect_model)

        # object annotator gizmo
        face_annotate = ObjectAnnotateGizmo(face_map)

        # gizmo to execute a chain of analyzers which count unknown faces and generate events and alerts
        alerts = AiAnalyzerGizmo(
            [
                unknown_face_event_detector,
                unknown_face_notifier,
            ]
        )

        # face crop gizmo
        face_extract = FaceExtractGizmo(
            target_image_size=reid_height,
            face_reid_map=face_map,
            reid_expiration_frames=reid_expiration_frames,
            zone_ids=[0],
            min_face_size=reid_height // 2,
        )

        # face ReID AI gizmo
        face_reid = AiSimpleGizmo(face_reid_model)

        # face reID search gizmo
        face_search = FaceSearchGizmo(
            face_map, self._open_db(), credence_count=credence_count
        )

        # display gizmo
        display: Union[VideoDisplayGizmo, VideoStreamerGizmo] = (
            VideoDisplayGizmo(stream_name, show_ai_overlay=True)
            if local_display
            else VideoStreamerGizmo(
                rtsp_url=f"rtsp://localhost:8554/{stream_name}",
                show_ai_overlay=True,
            )
        )

        watchdog = Watchdog(time_limit=20, tps_threshold=1, smoothing=0.95)
        face_detect.watchdog = watchdog  # attach watchdog

        #
        # define pipeline and run it
        #
        composition = Composition(
            source >> face_detect >> face_annotate >> alerts >> display,
            face_detect >> face_extract >> face_reid >> face_search,
        )
        composition.start(wait=False)

        return composition, watchdog
