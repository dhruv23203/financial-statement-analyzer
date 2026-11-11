"""Application context shared by all services (dependency injection).

Bundles settings, the database and the data provider so services never reach for
globals. Tests build a context with a temporary database and a fake provider.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.database.connection import Database
from app.providers import create_provider
from app.providers.base import FinancialDataProvider
from app.utils.logging_config import configure_logging
from config.settings import Settings, get_settings


@dataclass
class AppContext:
    settings: Settings
    db: Database
    provider_name: str
    _provider: FinancialDataProvider | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)

    @property
    def provider(self) -> FinancialDataProvider:
        if self._provider is None:
            self._provider = create_provider(self.provider_name, self.settings.sample_data_dir)
        return self._provider

    def close(self) -> None:
        self.db.dispose()

    def __enter__(self) -> "AppContext":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_context(settings: Settings | None = None, provider: str | None = None,
                  provider_instance: FinancialDataProvider | None = None,
                  initialise_db: bool = True,
                  sleep: Callable[[float], None] = time.sleep) -> AppContext:
    """Create directories, logging and the database; apply migrations if needed."""
    settings = settings or get_settings()
    settings.validate()
    settings.ensure_directories()
    configure_logging(settings.logs_dir, settings.log_level)
    db = Database(settings)
    if initialise_db:
        db.ensure_ready()
    name = provider or (provider_instance.name if provider_instance else settings.provider)
    return AppContext(settings=settings, db=db, provider_name=name,
                      _provider=provider_instance, sleep=sleep)
