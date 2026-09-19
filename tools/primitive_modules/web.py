from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def fetch_url(url: str, max_bytes: int = 262144, allow_private: bool = False) -> str:
    """Fetch one SSRF-safe URL and return bounded raw HTML/text with headers."""
    from .netutil import validate_public_url
    try:
        safe=validate_public_url(url,allow_private=bool(allow_private)); max_bytes=_bounded_int(max_bytes,1024,1048576)
        r=requests.get(safe,timeout=10,allow_redirects=False,stream=True,headers={"User-Agent":"AlAgent/1.0"})
        body=next(r.iter_content(chunk_size=max_bytes),b"")[:max_bytes]
        return _json({"url":safe,"status":r.status_code,"content_type":r.headers.get("content-type",""),"headers":dict(list(r.headers.items())[:50]),"body":body.decode("utf-8",errors="replace"),"truncated":len(body)>=max_bytes})
    except Exception as exc:return f"Error: fetch_url failed: {exc}"

def extract_readable_text(html: str, max_chars: int = 50000) -> str:
    """Extract readable text from bounded HTML without making a network request."""
    try:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(str(html)[:1048576],"html.parser")
        for tag in soup(["script","style","noscript","svg"]):tag.decompose()
        text="\n".join(x.strip() for x in soup.stripped_strings); max_chars=_bounded_int(max_chars,100,100000)
        return text[:max_chars] + ("\n[truncated]" if len(text)>max_chars else "")
    except Exception as exc:return f"Error: extract_readable_text failed: {exc}"

def extract_links(html: str, base_url: str = "", limit: int = 200) -> str:
    """Extract and resolve bounded links from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(str(html)[:1048576],"html.parser"); out=[]; seen=set()
        for a in soup.find_all("a",href=True):
            href=urljoin(base_url,str(a.get("href") or ""));
            if href in seen:continue
            seen.add(href); out.append({"url":href,"text":a.get_text(" ",strip=True)[:300]})
            if len(out)>=_bounded_int(limit,1,500):break
        return _json(out)
    except Exception as exc:return f"Error: extract_links failed: {exc}"

def extract_metadata(html: str, base_url: str = "") -> str:
    """Extract title, canonical, description, OpenGraph, and basic JSON-LD metadata from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(str(html)[:1048576],"html.parser"); meta={}
        for tag in soup.find_all("meta"):
            key=tag.get("property") or tag.get("name"); val=tag.get("content")
            if key and val and len(meta)<100:meta[str(key)]=str(val)[:2000]
        canonical=soup.find("link",rel=lambda v:v and "canonical" in v)
        return _json({"title":soup.title.get_text(" ",strip=True) if soup.title else "","canonical":urljoin(base_url,canonical.get("href")) if canonical and canonical.get("href") else "","metadata":meta})
    except Exception as exc:return f"Error: extract_metadata failed: {exc}"

def extract_images(html: str, base_url: str = "", limit: int = 100) -> str:
    """Extract bounded image URLs and alt text from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(str(html)[:1048576],"html.parser"); out=[]
        for img in soup.find_all("img"):
            src=img.get("src") or img.get("data-src")
            if src:out.append({"url":urljoin(base_url,str(src)),"alt":str(img.get("alt") or "")[:300]})
            if len(out)>=_bounded_int(limit,1,300):break
        return _json(out)
    except Exception as exc:return f"Error: extract_images failed: {exc}"

def extract_jsonld(html: str, limit: int = 20) -> str:
    """Extract bounded JSON-LD objects from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(str(html)[:1048576],"html.parser"); out=[]
        for tag in soup.find_all("script",attrs={"type":"application/ld+json"}):
            try:out.append(json.loads(tag.string or tag.get_text()))
            except Exception:continue
            if len(out)>=_bounded_int(limit,1,50):break
        return _json(out)
    except Exception as exc:return f"Error: extract_jsonld failed: {exc}"
