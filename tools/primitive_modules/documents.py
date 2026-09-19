from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def document_info(path: str) -> str:
    """Return document type/size and PDF page count when available."""
    try:
        p=_safe_workspace(path); payload={"path":str(p),"size":p.stat().st_size,"mime":mimetypes.guess_type(str(p))[0] or "application/octet-stream"}
        if p.suffix.lower()==".pdf" and shutil.which("pdfinfo"):
            proc=subprocess.run(["pdfinfo",str(p)],capture_output=True,text=True,timeout=10)
            for line in proc.stdout.splitlines():
                if ":" in line:
                    k,v=line.split(":",1); payload[k.strip().lower().replace(" ","_")]=v.strip()
        return _json(payload)
    except Exception as exc:return f"Error: document_info failed: {exc}"

def document_text(path: str, start_page: int = 1, end_page: int = 0, max_chars: int = 50000) -> str:
    """Extract bounded text from a workspace PDF or text document."""
    try:
        p=_safe_workspace(path); max_chars=_bounded_int(max_chars,100,MAX_TEXT)
        if p.suffix.lower()==".pdf":
            if not shutil.which("pdftotext"):return "Error: pdftotext is not installed."
            argv=["pdftotext","-layout"]
            if start_page>0:argv += ["-f",str(max(1,int(start_page)))]
            if end_page>0:argv += ["-l",str(max(int(start_page),int(end_page)))]
            argv += [str(p),"-"]
            proc=subprocess.run(argv,capture_output=True,text=True,timeout=30)
            if proc.returncode:return f"Error: pdftotext failed: {proc.stderr.strip()}"
            text=proc.stdout[:max_chars]
        else:text=p.read_text(encoding="utf-8",errors="replace")[:max_chars]
        return text + ("\n[truncated]" if len(text)>=max_chars else "")
    except Exception as exc:return f"Error: document_text failed: {exc}"

def render_document_page(path: str, page: int = 1, output: str = "") -> str:
    """Render one PDF page to a PNG inside the workspace."""
    try:
        p=_safe_workspace(path); page=max(1,int(page)); out=_safe_workspace(output or f"{p.stem}-page-{page}.png")
        if not shutil.which("pdftoppm"):return "Error: pdftoppm is not installed."
        prefix=str(out.with_suffix("")); proc=subprocess.run(["pdftoppm","-f",str(page),"-singlefile","-png","-r","110",str(p),prefix],capture_output=True,text=True,timeout=30)
        if proc.returncode:return f"Error: pdftoppm failed: {proc.stderr.strip()}"
        actual=Path(prefix+".png"); return _json({"path":str(actual),"page":page,"created":actual.exists()})
    except Exception as exc:return f"Error: render_document_page failed: {exc}"

def document_links(path: str, limit: int = 200) -> str:
    """Extract URL-like links from a workspace document's textual representation."""
    text=document_text(path,max_chars=100000)
    if text.startswith("Error:"):return text
    urls=list(dict.fromkeys(re.findall(r"https?://[^\s<>()\]}'\"]+",text)))[:_bounded_int(limit,1,500)]
    return _json(urls)

def document_images(path: str, limit: int = 100) -> str:
    """List embedded PDF image metadata using pdfimages without extracting files."""
    try:
        p=_safe_workspace(path)
        if p.suffix.lower()!=".pdf":return "Error: document_images currently supports PDF files."
        if not shutil.which("pdfimages"):return "Error: pdfimages is not installed."
        proc=subprocess.run(["pdfimages","-list",str(p)],capture_output=True,text=True,timeout=20)
        if proc.returncode:return f"Error: pdfimages failed: {proc.stderr.strip()}"
        return "\n".join(proc.stdout.splitlines()[:_bounded_int(limit,1,500)+2])
    except Exception as exc:return f"Error: document_images failed: {exc}"
