#
# object_storage_support.py: class to handle object storage manipulations
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements class for cloud object storage manipulations
#


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

    def upload_file_to_object_storage(
        self, file_path: str, bucket_name: str, object_name: str
    ):
        """
        Upload a file to cloud object storage bucket.

        Args:
            file_path: The local path of the file to upload
            bucket_name: The name of the bucket to upload to
            object_name: The name of the object (path within the bucket)

        """

        try:
            # Ensure the bucket exists
            if not self._client.bucket_exists(bucket_name):
                self._client.make_bucket(bucket_name)

            # Upload the file
            self._client.fput_object(bucket_name, object_name, file_path)

        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when uploading file '{file_path}' to '{bucket_name}/{object_name}: {e}"
            ) from e

    def download_file_from_object_storage(
        self,
        object_name: str,
        bucket_name: str,
        file_path: str,
    ):
        """
        Download a file from cloud object storage bucket.

        Args:
            object_name: The name of the object (path within the bucket)
            bucket_name: The name of the bucket to upload to
            file_path: The local path of the file to download to
        """

        try:
            self._client.fget_object(bucket_name, object_name, file_path)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when downloading file '{bucket_name}/{object_name} to '{file_path}': {e}"
            ) from e

    def delete_file_from_object_storage(
        self,
        object_name: str,
        bucket_name: str,
    ):
        """
        Delete a file from cloud object storage bucket.

        Args:
            object_name: The name of the object (path within the bucket)
            bucket_name: The name of the bucket to upload to
        """

        try:
            self._client.remove_object(bucket_name, object_name)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when deleting file '{bucket_name}/{object_name}': {e}"
            ) from e

    def check_file_exists_in_object_storage(
        self,
        object_name: str,
        bucket_name: str,
    ):
        """
        Checks if a file exists in cloud object storage bucket.

        Args:
            object_name: The name of the object (path within the bucket)
            bucket_name: The name of the bucket to upload to
        """

        try:
            return self._client.stat_object(bucket_name, object_name)
        except self._minio.S3Error as e:
            raise RuntimeError(
                f"Error occurred when deleting file '{bucket_name}/{object_name}': {e}"
            ) from e
