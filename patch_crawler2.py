import sys
content = open('app/ingestion/crawler_service.py', 'r', encoding='utf-8').read()

old_visited = 'if url in visited:\n                    queue.task_done()\n                    continue\n                visited.add(url)'
new_visited = 'from urllib.parse import urlparse, urlencode, parse_qs, urlunparse\n                def normalize_url(u):\n                    parsed = urlparse(u)\n                    # Strip trailing slashes and normalize case\n                    path = parsed.path.rstrip("/")\n                    if not path: path = "/"\n                    # Remove session-id/tracking bloat from queries to prevent infinite spider traps\n                    qs = parse_qs(parsed.query)\n                    for bad in ["session", "ref", "uid", "token", "source", "click"]:\n                        qs.pop(bad, None)\n                    query = urlencode(qs, doseq=True)\n                    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.params, query, ""))\n\n                norm_url = normalize_url(url)\n                if norm_url in visited:\n                    queue.task_done()\n                    continue\n                visited.add(norm_url)'

content = content.replace(old_visited, new_visited)

open('app/ingestion/crawler_service.py', 'w', encoding='utf-8').write(content)
