#
# object_storage_support.py: class to handle object storage manipulations
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements class for cloud object storage manipulations
#


import time
from datetime import timedelta
from .environment import import_optional_package
from dataclasses import dataclass


@dataclass
class ObjectStorageConfig:
    """
    Object storage configuration dataclass
    """

    endpoint: str  # The object storage endpoint URL
    access_key: str  # The access key for the cloud account
    secret_key: str  # The secret key for the cloud account
    bucket: str  # The name of the bucket to manage
    url_expiration_s: int = 3600  # The expiration time for the presigned URL in seconds

    def construct_direct_url(self, object_name: str):
        """
        Construct a URL to download a file from cloud object storage bucket.
        File may not exist in the bucket prior to this call.

        Args:
            object_name: The name of the object (path within the bucket)

        Returns:
            The URL to access the object
        """

        return f"{self.endpoint}/{self.bucket}/{object_name}"


class ObjectStorage:

    def __init__(self, config: ObjectStorageConfig):
        """
        Constructor

        Args:
            config: ObjectStorageConfig object containing object storage configuration
        """

        self._config = config

        self._minio = import_optional_package(
            "minio",
            custom_message="`minio` package is required for object storage uploads. "
            + "Please run `pip install degirum_tools[notifications]` to install required dependencies.",
        )

        self._client = self._minio.Minio(
            self._config.endpoint,
            access_key=self._config.access_key,
            secret_key=self._config.secret_key,
            secure=True,  # Set to False if not using HTTPS
        )

    def check_bucket_exits(self, retries=1):
        """
        Check if the bucket exists in cloud object storage
        Args:
            retries: Number of retries to check if the bucket exists
        """

        try:
            ret = False
            for _ in range(retries):
                ret = self._client.bucket_exists(self._config.bucket)
                if ret:
                    return True
                time.sleep(0.1)
            return ret

        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when checking if bucket '{self._config.bucket}' exists: {e}"
            ) from e

    def ensure_bucket_exists(self):
        """
        Ensure the bucket exists in cloud object storage
        """

        try:
            if not self._client.bucket_exists(self._config.bucket):
                self._client.make_bucket(self._config.bucket)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when ensuring bucket '{self._config.bucket}' exists: {e}"
            ) from e

    def list_bucket_contents(self):
        """
        List the contents of the bucket in cloud object storage

        Returns:
            List of objects in the bucket of None if the bucket does not exist
        """

        try:
            if self._client.bucket_exists(self._config.bucket):
                return self._client.list_objects(self._config.bucket, recursive=True)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when listing bucket '{self._config.bucket}': {e}"
            ) from e
        return None

    def delete_bucket_contents(self) -> bool:
        """
        Delete the bucket contents from cloud object storage

        Returns:
            True if bucket contents were deleted, False if bucket does not exist
        """

        try:
            if self._client.bucket_exists(self._config.bucket):
                objects = self._client.list_objects(self._config.bucket, recursive=True)
                for obj in objects:
                    self._client.remove_object(self._config.bucket, obj.object_name)
                return True
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when deleting bucket '{self._config.bucket}': {e}"
            )
        return False

    def delete_bucket(self):
        """
        Delete the bucket with all contents from cloud object storage
        """

        try:
            if self.delete_bucket_contents():
                self._client.remove_bucket(self._config.bucket)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when deleting bucket '{self._config.bucket}': {e}"
            )

    def generate_presigned_url(self, object_name: str):
        """
        Generate a presigned URL to download a file from cloud object storage bucket.
        File must exist in the bucket prior to this call.

        Args:
            object_name: The name of the object (path within the bucket)

        Returns:
            The presigned URL to download the object
        """

        try:
            return self._client.presigned_get_object(
                self._config.bucket,
                object_name,
                expires=timedelta(seconds=self._config.url_expiration_s),
            )
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when generating presigned URL for '{self._config.bucket}/{object_name}': {e}"
            ) from e

    def upload_file_to_object_storage(self, file_path: str, object_name: str):
        """
        Upload a file to cloud object storage bucket.

        Args:
            file_path: The local path of the file to upload
            object_name: The name of the object (path within the bucket)

        """

        try:
            # Upload the file
            self._client.fput_object(self._config.bucket, object_name, file_path)

        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when uploading file '{file_path}' to '{self._config.bucket}/{object_name}: {e}"
            ) from e

    def download_file_from_object_storage(
        self,
        object_name: str,
        file_path: str,
    ):
        """
        Download a file from cloud object storage bucket.

        Args:
            object_name: The name of the object (path within the bucket)
            file_path: The local path of the file to download to
        """

        try:
            self._client.fget_object(self._config.bucket, object_name, file_path)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when downloading file '{self._config.bucket}/{object_name} to '{file_path}': {e}"
            ) from e

    def delete_file_from_object_storage(
        self,
        object_name: str,
    ):
        """
        Delete a file from cloud object storage bucket.

        Args:
            object_name: The name of the object (path within the bucket)
        """

        try:
            self._client.remove_object(self._config.bucket, object_name)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when deleting file '{self._config.bucket}/{object_name}': {e}"
            ) from e

    def check_file_exists_in_object_storage(
        self,
        object_name: str,
    ):
        """
        Checks if a file exists in cloud object storage bucket.

        Args:
            object_name: The name of the object (path within the bucket)
        """

        try:
            return self._client.stat_object(self._config.bucket, object_name)
        except self._minio.S3Error:
            return None
