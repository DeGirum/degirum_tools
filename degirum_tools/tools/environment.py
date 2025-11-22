#
# environment.py: environment settings support
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements functions to operate with various environment settings
#

import dotenv, os, sys, importlib, types
from typing import Optional, Any

# environment variable names
var_TestMode = "TEST_MODE"
var_Token = "DEGIRUM_CLOUD_TOKEN"
var_CloudUrl = "DEGIRUM_CLOUD_PLATFORM_URL"
var_AiServer = "AISERVER_HOSTNAME_OR_IP"
var_CloudZoo = "CLOUD_ZOO_URL"
var_VideoSource = "CAMERA_ID"
var_AudioSource = "AUDIO_ID"
var_S3AccessKey = "S3_ACCESS_KEY"
var_S3SecretKey = "S3_SECRET_KEY"
var_MSTeamsTestWorkflowURL = "MSTEAMS_TEST_WORKFLOW_URL"


def reload_env(custom_file: str = "env.ini"):
    """Reload environment variables from file
    custom_file - name of the custom env file to try first;
        CWD, and ../CWD are searched for the file;
        if it is None or does not exist, `.env` file is loaded
    """

    if get_test_mode():
        return

    env_file = dotenv.find_dotenv(custom_file, usecwd=True)

    dotenv.load_dotenv(
        dotenv_path=env_file if env_file else None, override=True
    )  # load environment variables from file


def get_var(var: Optional[str], default_val: Any = None) -> Any:
    """Returns environment variable value"""
    if var is not None and var.isupper():  # treat `var` as env. var. name
        ret = os.getenv(var)
        if ret is None:
            if default_val is None:
                raise Exception(
                    f"Please define environment variable {var} in `.env` or `env.ini` file located in your CWD"
                )
            else:
                ret = default_val
    else:  # treat `var` literally
        ret = var
    return ret


def get_test_mode() -> bool:
    """Returns enable status of test mode from environment"""
    return bool(os.getenv(var_TestMode))


def get_token(default: Optional[str] = None) -> str:
    """Returns a token from .env file"""
    if in_colab():
        from google.colab import userdata

        return userdata.get("DEGIRUM_CLOUD_TOKEN")
    reload_env()  # reload environment variables from file
    return get_var(var_Token, default)


def get_ai_server_hostname() -> str:
    """Returns a AI server hostname/IP from .env file"""
    reload_env()  # reload environment variables from file
    return get_var(var_AiServer)


def get_cloud_zoo_url() -> str:
    """Returns a cloud zoo URL from .env file"""
    reload_env()  # reload environment variables from file

    cloud_url = "https://" + get_var(var_CloudUrl, "cs.degirum.com")
    zoo_url = get_var(var_CloudZoo, "")
    if zoo_url:
        cloud_url += "/" + zoo_url
    return cloud_url


def in_notebook() -> bool:
    """Returns `True` if the module is running in IPython kernel,
    `False` if in IPython shell or other Python shell.
    """
    return "ipykernel" in sys.modules


def in_colab() -> bool:
    """Returns `True` if the module is running in Google Colab environment"""
    return "google.colab" in sys.modules


def import_optional_package(
    pkg_name: str, is_long: bool = False, custom_message: Optional[str] = None
) -> types.ModuleType:
    """Import package with given name.
    Returns the package object.
    Raises error message if the package is not installed"""

    if is_long:
        print(f"Loading '{pkg_name}' package, be patient...")
    try:
        ret = importlib.import_module(pkg_name)
        if is_long:
            print(f"...done; '{pkg_name}' version: {ret.__version__}")
        return ret
    except ModuleNotFoundError as e:
        if custom_message:
            raise Exception(custom_message)
        else:
            raise Exception(
                f"\n*** Error loading '{pkg_name}' package: {e}. Not installed?\n"
            )


def configure_colab(
    *, video_file: Optional[str] = None, audio_file: Optional[str] = None
):
    """
    Configure Google Colab environment

    Args:
        video_file - path to video file to use instead of camera
        audio_file - path to wave file to use instead of microphone
    """

    # check if running under Colab
    if not in_colab():
        return

    import subprocess

    # define directories
    repo = "PySDKExamples"
    colab_root_dir = "/content"
    repo_dir = f"{colab_root_dir}/{repo}"
    work_dir = f"{repo_dir}/examples/workarea"

    if not os.path.exists(repo_dir):
        # request API token in advance
        def token_request():
            return input("\n\nEnter cloud API access token from cs.degirum.com:\n")

        token = token_request()

        def run_cmd(prompt, cmd):
            print(prompt + "... ", end="")
            result = subprocess.run(
                [cmd],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                print(result.stdout)
                raise Exception(f"{prompt} FAILS")
            print("DONE!")

        # clone PySDKExamples repo if not done yet
        os.chdir(colab_root_dir)
        run_cmd(
            "Cloning DeGirum/PySDKExamples repo",
            f"git clone https://github.com/DeGirum/{repo}",
        )

        # make repo root dir as CWD
        os.chdir(repo_dir)

        # install PySDKExamples requirements
        req_file = "requirements.txt"
        run_cmd(
            "Installing requirements (this will take a while)",
            f"pip install -r {req_file}",
        )

        # validate token
        print("Validating token...", end="")
        import degirum as dg

        while True:
            try:
                dg.connect(dg.CLOUD, "https://cs.degirum.com", token)
                break
            except Exception:
                print("\nProvided token is not valid!\n")
                token = token_request()
        print("DONE!")

        # configure env.ini
        env_file = "env.ini"
        print(f"Configuring {env_file} file...", end="")
        with open(env_file, "a") as file:
            file.write(f'{var_Token} = "{token}"\n')
            file.write(
                f'{var_VideoSource} = {video_file if video_file is not None else "../../images/example_video.mp4"}\n'
            )
            file.write(
                f'{var_AudioSource} = {audio_file if audio_file is not None else "../../images/example_audio.wav"}\n'
            )
        print("DONE!")

    # make working dir as CWD
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)


def to_valid_filename(in_str: str):
    """
    Convert string to a valid filename
    """
    import string, re

    valid_chars = "-_.()!@#$& %s%s" % (string.ascii_letters, string.digits)
    return re.sub("[^%s]+" % re.escape(valid_chars), "_", in_str)
