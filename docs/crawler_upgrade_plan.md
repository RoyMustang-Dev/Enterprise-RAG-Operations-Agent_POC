## Crawler Upgrade Plan (Safe, Incremental, Production‑Ready)

**Scope:** Upgrade only the crawler while keeping all other system components unchanged.

### Phase 1 — Safety & Correctness (Low Risk)
1. Replace blocking `requests` in robots with async `aiohttp`.
2. Use per‑domain robots cache and re‑fetch for subdomains.
3. Remove use of private `queue._unfinished_tasks` (replace with explicit counters).
4. Add retry with exponential backoff for `page.goto()` (2–3 attempts).
5. Add time budget + max URL cap (stop when exceeded).

### Phase 2 — Performance (Medium Risk)
1. Replace `SequenceMatcher` O(n²) with URL fingerprinting:
   - Normalize paths, hash patterns, store in a set.
2. Replace BeautifulSoup with `selectolax` for HTML parsing.
3. Add content hashing (`sha256`) for deduplication.
4. Use per‑domain concurrency limits (`asyncio.Semaphore`).
5. Add per‑worker `aiohttp.ClientSession` reuse.

### Phase 3 — Scalability (Optional / Enterprise)
1. Move queue from `asyncio.Queue` to Redis Streams.
2. Use distributed visited/content fingerprint sets in Redis.
3. Split HTTP‑first workers and browser workers into separate pools.
4. Add a persistent browser pool service (shared Chromium instances).
5. Store metadata in Postgres and raw content in S3.

### Phase 4 — Observability & QA
1. Add metrics: pages/sec, error rate, HTTP vs browser ratio, dedup ratio.
2. Add domain‑level crawl stats and throttling logs.
3. Add crawler regression suite with known URLs and validation checks.

### Acceptance Criteria
- No regression in current crawler output.
- 2–5x speed improvement from HTTP‑first path.
- Reduced duplicates and better structured markdown content.
- Observable stability across depth‑2 crawls.

