"""
Deprecated compatibility wrapper.

This module previously duplicated the SQLite implementation. It now forwards
to `app.infra.database` to avoid divergence. Replace imports with
`app.infra.database` directly when convenient.
"""
import warnings

from app.infra.database import (  # noqa: F401
    init_db,
    enable_wal,
    insert_page_async,
    get_all_pages,
    get_latest_session_data,
    init_ingestion_db,
    upsert_ingestion_job,
    get_ingestion_job,
)

warnings.warn(
    "app.storage.postgres is deprecated; use app.infra.database instead.",
    DeprecationWarning,
    stacklevel=2,
)
