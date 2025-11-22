#
# object_storage_support.py: class to handle object storage manipulations
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements class for cloud object storage manipulations
#

"""
Object Storage Support Module Overview
=====================================

Helper utilities for interacting with cloud object storage services such as
MinIO. The module also provides a lightweight local file-system backend for
testing without a remote service.

Key Classes:
    - ``ObjectStorageConfig``: Configuration parameters for object storage.
    - ``ObjectStorage``: Convenience wrapper around common bucket operations.

Typical Usage:
    1. Create an ``ObjectStorageConfig`` with connection parameters.
    2. Instantiate ``ObjectStorage`` using the configuration.
    3. Upload, download, or delete files using the instance methods.
"""


import time, os, shutil
import datetime
from typing import Optional
from dataclasses import dataclass
from .environment import import_optional_package


@dataclass
class ObjectStorageConfig:
    """Configuration for object storage connections.

    Attributes:
        endpoint (str): Object storage endpoint URL or local path.
        access_key (str): Access key for the storage account.
        secret_key (str): Secret key for the storage account.
        bucket (str): Bucket name or local directory name.
        url_expiration_s (int): Expiration time for presigned URLs.
    """

    endpoint: str  # The object storage endpoint URL or local path for local storage
    access_key: str  # The access key for the cloud account
    secret_key: str  # The secret key for the cloud account
    bucket: str  # The name of the bucket to manage or directory name for local storage
    url_expiration_s: int = 3600  # The expiration time for the presigned URL in seconds

    def construct_direct_url(self, object_name: str) -> str:
        """Construct a direct URL to an object.

        Args:
            object_name (str): Name of the object inside the bucket.
        """

        return f"{self.endpoint}/{self.bucket}/{object_name}"


class _LocalMinio:
    """Filesystem-backed mock of MinIO.

    Provides a subset of the MinIO API for unit tests and local development
    without requiring a running server.
    """

    @dataclass
    class Object:
        bucket_name: str
        object_name: str
        last_modified: datetime.datetime
        etag: str
        size: int
        content_type: Optional[str] = None
        is_dir: bool = False
        metadata: Optional[dict] = None
        version_id: Optional[str] = None

    def __init__(self, base_dir: str):
        """Initialize the local storage backend.

        Args:
            base_dir (str): Directory used to store all buckets. It will be
                created if it does not already exist.
        """
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def bucket_exists(self, bucket_name):
        """Check whether a bucket directory exists.

        Args:
            bucket_name (str): Name of the bucket to check.
        """
        bucket_path = os.path.join(self.base_dir, bucket_name)
        return os.path.isdir(bucket_path)

    def make_bucket(self, bucket_name):
        """Create a new bucket directory.

        Args:
            bucket_name (str): Name of the bucket to create.
        """
        bucket_path = os.path.join(self.base_dir, bucket_name)
        os.makedirs(bucket_path, exist_ok=True)

    def list_objects(self, bucket_name, prefix="", recursive=False):
        """Iterate over objects stored in a bucket.

        Args:
            bucket_name (str): Bucket name to inspect.
            prefix (str, optional): Only return objects that start with this
                prefix. Defaults to an empty string.
            recursive (bool, optional): If ``True`` search subdirectories
                recursively. Defaults to ``False``.
        """
        bucket_path = os.path.join(self.base_dir, bucket_name)
        if not os.path.isdir(bucket_path):
            raise FileNotFoundError(f"Bucket '{bucket_name}' does not exist.")

        for root, dirs, files in os.walk(bucket_path):
            # Clear subdirectories from the walk if recursion is disabled
            if not recursive:
                dirs.clear()
            for file in files:
                # Yield file paths relative to the bucket if they match the prefix
                if file.startswith(prefix):
                    yield _LocalMinio.Object(
                        bucket_name=bucket_name,
                        object_name=os.path.relpath(
                            os.path.join(root, file), bucket_path
                        ),
                        last_modified=datetime.datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(root, file)),
                            tz=datetime.timezone.utc,
                        ),
                        etag="",
                        size=os.path.getsize(os.path.join(root, file)),
                    )

    def remove_object(self, bucket_name, object_name):
        """Remove a specific object (file) from the bucket."""
        object_path = os.path.join(self.base_dir, bucket_name, object_name)
        if not os.path.isfile(object_path):
            raise FileNotFoundError(
                f"Object '{object_name}' does not exist in bucket '{bucket_name}'."
            )
        os.remove(object_path)

    def remove_bucket(self, bucket_name):
        """Remove an empty bucket (directory)"""
        bucket_path = os.path.join(self.base_dir, bucket_name)
        if not os.path.isdir(bucket_path):
            raise FileNotFoundError(f"Bucket '{bucket_name}' does not exist.")
        if os.listdir(bucket_path):
            # Prevent removal if the bucket is not empty
            raise OSError(f"Bucket '{bucket_name}' is not empty.")
        os.rmdir(bucket_path)

    def presigned_get_object(
        self, bucket_name, object_name, expires=datetime.timedelta(hours=1)
    ):
        """Return a simulated presigned URL.

        For the local backend this simply returns the path to ``object_name``
        inside ``bucket_name``. The ``expires`` parameter is ignored.
        """
        return os.path.join(self.base_dir, bucket_name, object_name)

    def fput_object(self, bucket_name, object_name, file_path):
        """Upload a file to the specified bucket and object path."""
        bucket_path = os.path.join(self.base_dir, bucket_name)
        if not os.path.isdir(bucket_path):
            raise FileNotFoundError(f"Bucket '{bucket_name}' does not exist.")
        dest_path = os.path.join(bucket_path, object_name)
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(file_path, dest_path)

    def fget_object(self, bucket_name, object_name, file_path):
        """Download an object (file) from the bucket to the local filesystem."""
        object_path = os.path.join(self.base_dir, bucket_name, object_name)
        if not os.path.isfile(object_path):
            raise FileNotFoundError(
                f"Object '{object_name}' does not exist in bucket '{bucket_name}'."
            )
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        shutil.copy2(object_path, file_path)

    def stat_object(self, bucket_name, object_name):
        """Retrieve metadata for a specific object."""
        object_path = os.path.join(self.base_dir, bucket_name, object_name)
        if not os.path.isfile(object_path):
            raise FileNotFoundError(
                f"Object '{object_name}' does not exist in bucket '{bucket_name}'."
            )
        stat = os.stat(object_path)
        # Return a dictionary with size and last modification time
        return {
            "size": stat.st_size,
            "last_modified": datetime.datetime.fromtimestamp(stat.st_mtime),
        }


class ObjectStorage:
    """Convenience wrapper around common object storage operations.

    This helper abstracts interaction with either a real MinIO server or a
    local directory acting as an object store. It exposes simple methods for
    bucket management and file uploads/downloads.
    """

    def __init__(self, config: ObjectStorageConfig):
        """Initialize the storage helper.

        Depending on ``config.endpoint`` this helper connects either to a real
        MinIO server or to a local directory used as a mock object store.

        Args:
            config (ObjectStorageConfig): Storage configuration.
        """

        self._config = config

        if os.path.exists(config.endpoint):
            self._client = _LocalMinio(config.endpoint)
        else:
            minio = import_optional_package(
                "minio",
                custom_message="`minio` package is required for object storage uploads. "
                + "Please run `pip install degirum_tools[notifications]` to install required dependencies.",
            )

            self._client = minio.Minio(
                self._config.endpoint,
                access_key=self._config.access_key,
                secret_key=self._config.secret_key,
                secure=True,  # Set to False if not using HTTPS
            )

    def check_bucket_exits(self, retries=1):
        """Check whether the configured bucket exists.

        Args:
            retries (int, optional): Number of retry attempts.

        Returns:
            bool (bool): ``True`` if the bucket exists.
        """

        try:
            ret = False
            for _ in range(retries):
                ret = self._client.bucket_exists(self._config.bucket)
                if ret:
                    return True
                time.sleep(0.1)
            return ret

        except Exception as e:
            raise RuntimeError(
                f"Error occurred when checking if bucket '{self._config.bucket}' exists: {e}"
            ) from e

    def ensure_bucket_exists(self):
        """Create the bucket if it does not exist.

        Raises:
            RuntimeError: If bucket creation fails.
        """

        try:
            if not self._client.bucket_exists(self._config.bucket):
                self._client.make_bucket(self._config.bucket)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when ensuring bucket '{self._config.bucket}' exists: {e}"
            ) from e

    def list_bucket_contents(self):
        """List objects in the bucket.

        Returns:
            (Iterable or None): Iterator over objects, or ``None`` if the bucket does not exist.

        Raises:
            RuntimeError: If listing fails.
        """

        try:
            if self._client.bucket_exists(self._config.bucket):
                return self._client.list_objects(self._config.bucket, recursive=True)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when listing bucket '{self._config.bucket}': {e}"
            ) from e
        return None

    def delete_bucket_contents(self) -> bool:
        """Remove all objects from the bucket."""

        try:
            if self._client.bucket_exists(self._config.bucket):
                objects = self._client.list_objects(self._config.bucket, recursive=True)
                for obj in objects:
                    self._client.remove_object(self._config.bucket, obj.object_name)
                return True
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when deleting bucket '{self._config.bucket}': {e}"
            )
        return False

    def delete_bucket(self):
        """Delete the bucket and all of its objects.

        Raises:
            RuntimeError: If the bucket cannot be removed.
        """

        try:
            if self.delete_bucket_contents():
                self._client.remove_bucket(self._config.bucket)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when deleting bucket '{self._config.bucket}': {e}"
            )

    def generate_presigned_url(self, object_name: str):
        """Return a presigned download URL for an object.

        Args:
            object_name (str): Name of the object within the bucket.

        Returns:
            (str): Temporary download URL for the object.
        """

        try:
            return self._client.presigned_get_object(
                self._config.bucket,
                object_name,
                expires=datetime.timedelta(seconds=self._config.url_expiration_s),
            )
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when generating presigned URL for '{self._config.bucket}/{object_name}': {e}"
            ) from e

    def upload_file_to_object_storage(self, file_path: str, object_name: str):
        """Upload a file to the configured bucket.

        Args:
            file_path (str): Path of the local file to upload.
            object_name (str): Name of the object within the bucket.

        Raises:
            RuntimeError: If the upload fails.
        """

        try:
            # Upload the file
            self._client.fput_object(self._config.bucket, object_name, file_path)

        except Exception as e:
            raise RuntimeError(
                f"Error occurred when uploading file '{file_path}' to '{self._config.bucket}/{object_name}: {e}"
            ) from e

    def download_file_from_object_storage(
        self,
        object_name: str,
        file_path: str,
    ):
        """Download a file from the configured bucket.

        Args:
            object_name (str): Name of the object within the bucket.
            file_path (str): Local path where the file will be saved.

        Raises:
            RuntimeError: If the download fails.
        """

        try:
            self._client.fget_object(self._config.bucket, object_name, file_path)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when downloading file '{self._config.bucket}/{object_name} to '{file_path}': {e}"
            ) from e

    def delete_file_from_object_storage(
        self,
        object_name: str,
    ):
        """Delete a file from the configured bucket.

        Args:
            object_name (str): Name of the object within the bucket.

        Raises:
            RuntimeError: If the deletion fails.
        """

        try:
            self._client.remove_object(self._config.bucket, object_name)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred when deleting file '{self._config.bucket}/{object_name}': {e}"
            ) from e

    def check_file_exists_in_object_storage(
        self,
        object_name: str,
    ):
        """Check whether a file exists in the configured bucket.

        Args:
            object_name (str): Name of the object within the bucket.
        """

        try:
            return self._client.stat_object(self._config.bucket, object_name)
        except Exception:
            return None
