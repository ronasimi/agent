from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def image_info(path: str) -> str:
    """Return deterministic image dimensions/format metadata without visual interpretation."""
    try:
        from PIL import Image
        p=_safe_workspace(path)
        with Image.open(p) as img:return _json({"path":str(p),"width":img.width,"height":img.height,"format":img.format,"mode":img.mode,"bytes":p.stat().st_size})
    except ImportError:return "Error: Pillow is not installed."
    except Exception as exc:return f"Error: image_info failed: {exc}"

def media_info(path: str) -> str:
    """Return deterministic media metadata using ffprobe when available, otherwise MIME/size metadata."""
    try:
        p=_safe_workspace(path)
        if shutil.which("ffprobe"):
            proc=subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_format","-show_streams",str(p)],capture_output=True,text=True,timeout=15)
            if proc.returncode==0:return proc.stdout
        return _json({"path":str(p),"size":p.stat().st_size,"mime":mimetypes.guess_type(str(p))[0] or "application/octet-stream"})
    except Exception as exc:return f"Error: media_info failed: {exc}"

def image_resize(path: str, output: str, max_width: int = 1600, max_height: int = 1600) -> str:
    """Resize an image into a new workspace file while preserving aspect ratio."""
    try:
        from PIL import Image
        src=_safe_workspace(path); dst=_safe_workspace(output); dst.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(src) as img:
            img.thumbnail((_bounded_int(max_width,16,8192),_bounded_int(max_height,16,8192))); img.save(dst)
            return _json({"path":str(dst),"width":img.width,"height":img.height})
    except Exception as exc:return f"Error: image_resize failed: {exc}"

def image_crop(path: str, output: str, left: int, top: int, right: int, bottom: int) -> str:
    """Crop an image into a new workspace file."""
    try:
        from PIL import Image
        src=_safe_workspace(path); dst=_safe_workspace(output); dst.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(src) as img:
            box=(int(left),int(top),int(right),int(bottom)); cropped=img.crop(box); cropped.save(dst); return _json({"path":str(dst),"width":cropped.width,"height":cropped.height})
    except Exception as exc:return f"Error: image_crop failed: {exc}"

def image_convert(path: str, output: str, format: str = "PNG") -> str:
    """Convert an image into a new workspace file using PNG, JPEG, or WEBP."""
    try:
        from PIL import Image
        fmt=format.upper()
        if fmt not in {"PNG","JPEG","WEBP"}:return "Error: format must be PNG, JPEG, or WEBP."
        src=_safe_workspace(path); dst=_safe_workspace(output); dst.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(src) as img:img.save(dst,format=fmt)
        return _json({"path":str(dst),"format":fmt})
    except Exception as exc:return f"Error: image_convert failed: {exc}"
