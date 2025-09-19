# remote_assets.py: list of DeGirum remote media assets - images and videos
# Copyright DeGirum Corporation 2025
# All rights reserved
# Defines module global variables for remote media assets containing URLs to images and videos


def _discover_media_files():

    import re
    import os
    import requests

    # List of repositories to pull from
    REPOS = [
        {
            "owner": "DeGirum",
            "repo": "PySDKExamples",
            "path": "images",
            "branch": "main",
        },
        # add more repos here...
    ]

    IMAGE_EXTS = {".jpg"}
    VIDEO_EXTS = {".mp4"}

    # Precompiled regex patterns (shared)
    _CAMEL_RE1 = re.compile(r"(.)([A-Z][a-z]+)")
    _CAMEL_RE2 = re.compile(r"([a-z0-9])([A-Z])")
    _INVALID_CHARS_RE = re.compile(r"[^0-9a-zA-Z_]")
    _STARTS_WITH_DIGIT_RE = re.compile(r"^[0-9]")

    def _camel_to_snake(name):
        # Step 1: CamelCase → snake_case
        s1 = _CAMEL_RE1.sub(r"\1_\2", name)
        snake = _CAMEL_RE2.sub(r"\1_\2", s1).lower()

        # Step 2: Sanitize to valid Python identifier
        snake = _INVALID_CHARS_RE.sub("_", snake)  # replace invalid chars
        if _STARTS_WITH_DIGIT_RE.match(snake):  # prefix if starts with digit
            snake = f"_{snake}"
        return snake

    def _list_media_files(owner, repo, path, branch):
        api_url = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}/contents/{path}"
            f"?ref={branch}"
        )
        resp = requests.get(api_url, timeout=3)
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            raise RuntimeError(
                f"Unexpected API response structure for {owner}/{repo}/{path}"
            )
        media_items = []
        for item in items:
            name = item.get("name", "")
            _, ext = os.path.splitext(name)
            ext = ext.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                download_url = item.get("download_url")
                if download_url:
                    media_items.append(
                        {
                            "name": name,
                            "url": download_url,
                            "ext": ext,
                        }
                    )
        return media_items

    # Define the RemoteMedia class
    class RemoteMedia(str):
        __slots__ = ("_ext",)
        _ext: str  # type hint for mypy

        def __new__(cls, value, ext):
            obj = super().__new__(cls, value)
            obj._ext = ext.lower()
            return obj

        @property
        def is_image(self) -> bool:
            return self._ext in IMAGE_EXTS

        @property
        def is_video(self) -> bool:
            return self._ext in VIDEO_EXTS

        @property
        def kind(self) -> str:
            if self.is_image:
                return "image"
            elif self.is_video:
                return "video"
            return "other"

    media: dict = {}
    errors = []

    for repo_info in REPOS:
        try:
            media_files = _list_media_files(**repo_info)
        except Exception as e:
            errors.append(str(e))
            continue

        for item in media_files:
            try:
                stem, _ = os.path.splitext(item["name"])
                varname = _camel_to_snake(stem)

                # Ensure unique variable name
                original = varname
                i = 1
                while varname in globals() or varname in media:
                    varname = f"{original}_{i}"
                    i += 1

                # Use RemoteMedia as the value
                media[varname] = RemoteMedia(item["url"], item["ext"])

            except Exception as e:
                errors.append(str(e))
                continue

    if errors:
        media["__errors"] = errors

    return media


# remote_assets module global variable
_assets: dict = {}


def list_images():
    """Return a dictionary of image media assets."""
    return {k: v for k, v in _assets.items() if v.is_image}


def list_videos():
    """Return a dictionary of video media assets."""
    return {k: v for k, v in _assets.items() if v.is_video}


# filter dir() output to only include media keys
def __dir__():
    return _assets.keys()


# Inject into module namespace
_assets = _discover_media_files()
globals().update(_assets)
