"""Primitive compatibility recipes for high-level web/document tools."""
from __future__ import annotations

P = lambda default, description: {"default": default, "description": description}

RECIPE_SPECS = [
    {
        "key":"grounding.weather.current_forecast","version":2,"name":"weather.current_forecast","target_tool":"weather_grounding",
        "description":"Ground current weather/forecast retrieval with keyless structured geocoding and forecast data; web discovery remains an independent fallback.","tags":["weather","forecast","grounding","structured","open-meteo"],
        "parameters":{
            "location":P("London, Ontario, Canada","Requested or explicitly stored place name"),
            "forecast_days":P(8,"Forecast horizon including today, bounded to 1-16 days"),
            "query":P("current weather forecast","Fallback weather query including the requested or recalled location"),
        },
        "pipeline":[
            {"id":"place","tool":"geocode_location","args":{"query":{"$param":"location"},"count":1}},
            {"id":"forecast","tool":"weather_forecast","args":{"latitude":{"$ref":"place","path":"0.latitude"},"longitude":{"$ref":"place","path":"0.longitude"},"forecast_days":{"$param":"forecast_days","default":8},"timezone_name":"auto"}},
            {"id":"result","tool":"compose_object","args":{"data":{"location":{"$param":"location"},"place":{"$ref":"place","path":"0"},"forecast":{"$ref":"forecast"}}}},
        ],
    },
    {
        "key":"compat.browse_url","version":1,"name":"compat.browse_url","target_tool":"browse_url",
        "description":"Fetch a public page and extract readable text without sending raw HTML through the model.","tags":["compat","browse_url","web","readable","page"],
        "parameters":{"url":P("https://example.com","Public HTTP(S) URL"),"max_chars":P(20000,"Maximum readable text characters")},
        "pipeline":[
            {"id":"page","tool":"fetch_url","args":{"url":{"$param":"url","default":"https://example.com"},"max_bytes":262144,"allow_private":False}},
            {"id":"text","tool":"extract_readable_text","args":{"html":{"$ref":"page","path":"body"},"max_chars":{"$param":"max_chars","default":20000}}},
            {"id":"result","tool":"compose_object","args":{"data":{"url":{"$ref":"page","path":"url"},"content_type":{"$ref":"page","path":"content_type"},"content":{"$ref":"text"}}}},
        ],
    },
    {
        "key":"compat.page_metadata","version":1,"name":"compat.page_metadata","target_tool":"page_metadata",
        "description":"Fetch a public page once and extract title, canonical, and metadata from the fetched HTML.","tags":["compat","page_metadata","web","metadata"],
        "parameters":{"url":P("https://example.com","Public HTTP(S) URL")},
        "pipeline":[
            {"id":"page","tool":"fetch_url","args":{"url":{"$param":"url","default":"https://example.com"},"max_bytes":524288,"allow_private":False}},
            {"id":"meta","tool":"extract_metadata","args":{"html":{"$ref":"page","path":"body"},"base_url":{"$param":"url","default":"https://example.com"}}},
            {"id":"result","tool":"compose_object","args":{"data":{"url":{"$ref":"page","path":"url"},"content_type":{"$ref":"page","path":"content_type"},"metadata":{"$ref":"meta"}}}},
        ],
    },
    {
        "key":"compat.page_links","version":1,"name":"compat.page_links","target_tool":"page_links",
        "description":"Fetch once, extract links, then deterministically filter/deduplicate them by domain.","tags":["compat","page_links","web","links"],
        "parameters":{"url":P("https://example.com","Public HTTP(S) URL"),"same_domain":P(True,"Restrict links to source host"),"limit":P(50,"Maximum links")},
        "pipeline":[
            {"id":"page","tool":"fetch_url","args":{"url":{"$param":"url","default":"https://example.com"},"max_bytes":524288,"allow_private":False}},
            {"id":"links","tool":"extract_links","args":{"html":{"$ref":"page","path":"body"},"base_url":{"$param":"url","default":"https://example.com"},"limit":200}},
            {"id":"result","tool":"filter_links","args":{"links":{"$ref":"links"},"base_url":{"$param":"url","default":"https://example.com"},"same_domain":{"$param":"same_domain","default":True},"limit":{"$param":"limit","default":50}}},
        ],
    },
    {
        "key":"compat.read_feed","version":2,"name":"compat.read_feed","target_tool":"read_feed",
        "description":"Fetch RSS/Atom XML and parse the already-fetched body into bounded feed entries.","tags":["compat","rss","atom","feed"],
        "parameters":{"url":P("https://feeds.bbci.co.uk/news/rss.xml","Feed URL"),"limit":P(20,"Maximum entries")},
        "pipeline":[
            {"id":"feed","tool":"fetch_url","args":{"url":{"$param":"url","default":"https://feeds.bbci.co.uk/news/rss.xml"},"max_bytes":1048576,"allow_private":False}},
            {"id":"result","tool":"parse_feed","args":{"xml_text":{"$ref":"feed","path":"body"},"url":{"$ref":"feed","path":"url"},"limit":{"$param":"limit","default":20}}},
        ],
    },
    {
        "key":"compat.page_fingerprint","version":1,"name":"compat.page_fingerprint","target_tool":"page_fingerprint",
        "description":"Fetch readable page text, hash it, count it, and return a bounded sample without persisting state.","tags":["compat","page","fingerprint","hash","monitor"],
        "parameters":{"url":P("https://example.com","Public HTTP(S) URL")},
        "pipeline":[
            {"id":"page","tool":"fetch_url","args":{"url":{"$param":"url","default":"https://example.com"},"max_bytes":524288,"allow_private":False}},
            {"id":"text","tool":"extract_readable_text","args":{"html":{"$ref":"page","path":"body"},"max_chars":60000}},
            {"id":"hash","tool":"hash_text","args":{"text":{"$ref":"text"},"algorithm":"sha256"}},
            {"id":"count","tool":"text_count","args":{"text":{"$ref":"text"}}},
            {"id":"sample","tool":"text_head","args":{"text":{"$ref":"text"},"lines":20}},
            {"id":"result","tool":"compose_object","args":{"data":{"url":{"$ref":"page","path":"url"},"sha256":{"$ref":"hash","path":"digest","default":""},"stats":{"$ref":"count"},"sample":{"$ref":"sample"}}}},
        ],
    },
    {
        "key":"compat.extract_document.local","version":1,"name":"compat.extract_document.local","target_tool":"extract_document",
        "description":"Local-file subset of extract_document: inspect a workspace document and extract bounded text/pages using document primitives.","tags":["compat","extract_document","document","local","partial"],
        "parameters":{"path_or_url":P("document.pdf","Workspace-local document path"),"max_pages":P(30,"Maximum PDF pages"),"max_chars":P(30000,"Maximum text characters")},
        "pipeline":[
            {"id":"info","tool":"document_info","args":{"path":{"$param":"path_or_url","default":"document.pdf"}}},
            {"id":"text","tool":"document_text","args":{"path":{"$param":"path_or_url","default":"document.pdf"},"start_page":1,"end_page":{"$param":"max_pages","default":30},"max_chars":{"$param":"max_chars","default":30000}}},
            {"id":"result","tool":"compose_object","args":{"data":{"info":{"$ref":"info"},"text":{"$ref":"text"}}}},
        ],
        "coverage":"partial",
    },
]

NATIVE_ONLY = {
    "news_search": "news metasearch is already a narrow external-source primitive and has no lower local composition",
    "web_search": "search-engine discovery is already a narrow external-source primitive and has no lower local composition",
    "wiki_search": "specialized external search endpoint; no smaller local composition",
    "discover_site": "bounded sitemap/robots traversal requires a queue/recursive fetch loop beyond current recipe semantics",
    "page_diff": "persists monitoring state and is mutating by policy, while dynamic recipes are read-only",
    "take_web_screenshot": "browser/media artifact creation is side-effecting and requires Playwright",
    "attach_media": "model-media bridge is a frontend/runtime capability, not deterministic data composition",
    "generate_pdf_report": "creates an artifact and is intentionally mutating",
    "search_packages": "package-manager discovery is already a narrow external query",
    "install_package": "mutating package installation is excluded from recipes",
}
