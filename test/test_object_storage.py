#
# test_object_storage.py: unit tests for ObjectStorage class
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests for ObjectStorage class - object storage manipulations
#

import os
import tempfile


def test_object_storage():
    """
    Test for ObjectStorage class
    """

    import degirum_tools

    AWS_S3_ACCESS_KEY = os.getenv("AWS_S3_ACCESS_KEY")
    AWS_S3_SECRET_KEY = os.getenv("AWS_S3_SECRET_KEY")
    if AWS_S3_ACCESS_KEY is None or AWS_S3_SECRET_KEY is None:
        print(
            "AWS_S3_ACCESS_KEY and/or AWS_S3_SECRET_KEY environment variables are not set: test_object_storage is skipped"
        )
        return  # Skip test

    cfg = degirum_tools.ObjectStorageConfig(
        endpoint="s3.us-west-1.amazonaws.com",
        access_key=AWS_S3_ACCESS_KEY,
        secret_key=AWS_S3_SECRET_KEY,
        bucket="dg-degirum-tools-test-s3",
    )

    obj_storage = degirum_tools.ObjectStorage(cfg)

    obj_storage.ensure_bucket_exists()
    assert obj_storage.check_bucket_exits(100)

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = temp_dir + "/test_file.txt"
        object_name = "test_object.txt"
        contents = "Hello, World!"

        with open(local_path, "w") as f:
            f.write(contents)

        assert obj_storage.check_file_exists_in_object_storage(object_name) is None
        obj_storage.upload_file_to_object_storage(local_path, object_name)
        assert obj_storage.check_file_exists_in_object_storage(object_name) is not None

        downloaded_local_path = temp_dir + "/" + object_name
        obj_storage.download_file_from_object_storage(
            object_name, downloaded_local_path
        )
        assert os.path.exists(downloaded_local_path)
        with open(downloaded_local_path, "r") as f:
            assert f.read() == contents

        obj_storage.delete_file_from_object_storage(object_name)
        assert obj_storage.check_file_exists_in_object_storage(object_name) is None
