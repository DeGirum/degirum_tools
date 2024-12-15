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
import pytest


def test_object_storage(s3_credentials):
    """
    Test for ObjectStorage class
    """

    import degirum_tools

    def test_config(cfg):

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
            assert (
                obj_storage.check_file_exists_in_object_storage(object_name) is not None
            )

            downloaded_local_path = temp_dir + "/" + object_name
            obj_storage.download_file_from_object_storage(
                object_name, downloaded_local_path
            )
            assert os.path.exists(downloaded_local_path)
            with open(downloaded_local_path, "r") as f:
                assert f.read() == contents

            obj_storage.delete_file_from_object_storage(object_name)
            assert obj_storage.check_file_exists_in_object_storage(object_name) is None

    # test with S3
    s3_cfg = degirum_tools.ObjectStorageConfig(**s3_credentials)
    if s3_cfg.access_key is None or s3_cfg.secret_key is None:
        pytest.skip(
            "S3_ACCESS_KEY and/or S3_SECRET_KEY environment variables are not set"
        )
    test_config(s3_cfg)

    # local test
    with tempfile.TemporaryDirectory() as temp_dir:
        local_cfg = degirum_tools.ObjectStorageConfig(
            endpoint=temp_dir,
            access_key="",
            secret_key="",
            bucket="test_bucket",
        )
        test_config(local_cfg)
