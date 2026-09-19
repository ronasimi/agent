from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def fetch_url(url: str, max_bytes: int = 262144, allow_private: bool = False) -> str:
    """Fetch one SSRF-safe URL with bounded redirect validation and return raw text plus headers."""
    from ..netutil import fetch_bytes
    try:
        max_bytes=_bounded_int(max_bytes,1024,1048576)
        final, response, body=fetch_bytes(url,timeout=10,max_bytes=max_bytes,max_redirects=3,allow_private=bool(allow_private))
        encoding=response.encoding or "utf-8"
        try:text=body.decode(encoding,errors="replace")
        except LookupError:text=body.decode("utf-8",errors="replace")
        return _json({"url":final,"status":response.status_code,"content_type":response.headers.get("content-type",""),"headers":dict(list(response.headers.items())[:50]),"body":text,"truncated":False})
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

def filter_links(links: list, base_url: str = "", same_domain: bool = True, limit: int = 50) -> str:
    """Filter/deduplicate extracted HTTP(S) link objects and optionally restrict them to the base URL hostname."""
    origin=(urlparse(base_url).hostname or "").lower(); out=[]; seen=set()
    for item in links or []:
        if not isinstance(item,dict):continue
        raw=str(item.get("url") or ""); parsed=urlparse(raw)
        if parsed.scheme not in {"http","https"} or not parsed.hostname:continue
        same=(parsed.hostname or "").lower()==origin if origin else True
        if same_domain and not same:continue
        clean=parsed._replace(fragment="").geturl()
        if clean in seen:continue
        seen.add(clean); out.append({"url":clean,"text":str(item.get("text") or "")[:300],"same_domain":same})
        if len(out)>=_bounded_int(limit,1,200):break
    return _json({"source":base_url,"links":out})


def parse_feed(xml_text: str, url: str = "", limit: int = 20) -> str:
    """Parse bounded RSS/Atom XML already fetched by another pipeline stage."""
    try:
        import feedparser
        feed=feedparser.parse(str(xml_text)[:2097152]); entries=[]
        try:
            from bs4 import BeautifulSoup
        except Exception:
            BeautifulSoup=None
        for item in feed.entries[:_bounded_int(limit,1,50)]:
            summary=str(item.get("summary") or item.get("description") or "")
            if BeautifulSoup is not None:summary=BeautifulSoup(summary,"html.parser").get_text(" ",strip=True)
            entries.append({"title":str(item.get("title") or "")[:500],"url":str(item.get("link") or "")[:2000],"published":str(item.get("published") or item.get("updated") or "")[:200],"author":str(item.get("author") or "")[:300],"summary":summary[:1500]})
        return _json({"url":url,"feed_title":str(feed.feed.get("title") or "")[:500],"entries":entries})
    except Exception as exc:return f"Error: parse_feed failed: {exc}"

def fetch_json(url: str, max_bytes: int = 262144, allow_private: bool = False) -> str:
    """Fetch one SSRF-validated JSON URL with bounded redirects and decode the JSON body."""
    from ..netutil import fetch_bytes
    try:
        max_bytes=_bounded_int(max_bytes,1024,1048576)
        final, response, body=fetch_bytes(url,timeout=10,max_bytes=max_bytes,max_redirects=3,allow_private=bool(allow_private))
        return _json(json.loads(body.decode(response.encoding or "utf-8",errors="replace")))
    except Exception as exc:return f"Error: fetch_json failed: {exc}"


def extract_tables(html: str, limit: int = 10, max_rows: int = 50) -> str:
    """Extract bounded HTML tables into captions, headers, and cell rows."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(str(html)[:1048576], "html.parser")
        limit = _bounded_int(limit, 1, 30); max_rows = _bounded_int(max_rows, 1, 500); tables = []
        for table in soup.find_all("table")[:limit]:
            caption = table.find("caption")
            rows = []
            for tr in table.find_all("tr")[:max_rows]:
                cells = [cell.get_text(" ", strip=True)[:2000] for cell in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            first = table.find("tr")
            headers = [cell.get_text(" ", strip=True)[:500] for cell in first.find_all("th")] if first else []
            tables.append({"caption": caption.get_text(" ", strip=True)[:500] if caption else "", "headers": headers, "rows": rows, "truncated": len(table.find_all("tr")) > max_rows})
        return _json({"tables": tables, "count": len(tables)})
    except Exception as exc:
        return f"Error: extract_tables failed: {exc}"
