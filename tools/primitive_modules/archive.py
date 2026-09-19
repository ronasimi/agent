from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def archive_list(path: str, limit: int = 500) -> str:
    """List members of ZIP/TAR archives in the workspace without extracting them."""
    import tarfile, zipfile
    try:
        p=_safe_workspace(path); limit=_bounded_int(limit,1,2000); rows=[]
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                rows=[{"name":i.filename,"size":i.file_size,"compressed":i.compress_size} for i in z.infolist()[:limit]]
        elif tarfile.is_tarfile(p):
            with tarfile.open(p) as t: rows=[{"name":i.name,"size":i.size,"type":"dir" if i.isdir() else "file"} for i in t.getmembers()[:limit]]
        else:return "Error: unsupported archive format."
        return _json({"path":str(p),"members":rows})
    except Exception as exc:return f"Error: archive_list failed: {exc}"

def archive_extract(path: str, destination: str, members: list[str] | None = None, max_files: int = 200) -> str:
    """Safely extract selected ZIP/TAR members into a workspace directory; path traversal is blocked."""
    import tarfile, zipfile
    try:
        src=_safe_workspace(path); dst=_safe_workspace(destination); dst.mkdir(parents=True,exist_ok=True); selected=set(members or []); max_files=_bounded_int(max_files,1,500); written=[]
        def safe_target(name):
            target=(dst/name).resolve()
            if os.path.commonpath([str(dst.resolve()),str(target)])!=str(dst.resolve()):raise ValueError(f"unsafe archive path: {name}")
            return target
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as z:
                for info in z.infolist():
                    if selected and info.filename not in selected:continue
                    target=safe_target(info.filename)
                    if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as r,target.open("wb") as w:shutil.copyfileobj(r,w,length=1024*1024)
                    written.append(str(target))
                    if len(written)>=max_files:break
        elif tarfile.is_tarfile(src):
            with tarfile.open(src) as t:
                for info in t.getmembers():
                    if selected and info.name not in selected:continue
                    if not info.isfile():continue
                    target=safe_target(info.name); target.parent.mkdir(parents=True,exist_ok=True); source=t.extractfile(info)
                    if source:
                        with source,target.open("wb") as w:shutil.copyfileobj(source,w,length=1024*1024)
                    written.append(str(target))
                    if len(written)>=max_files:break
        else:return "Error: unsupported archive format."
        return _json({"destination":str(dst),"files":written,"count":len(written)})
    except Exception as exc:return f"Error: archive_extract failed: {exc}"
