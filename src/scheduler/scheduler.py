"""APScheduler Task Orchestrator for Daily Market-Close Automated Execution."""

from __future__ import annotations
import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import AppConfig
from src.scheduler.tasks import run_full_pipeline_task


class PipelineScheduler:
    """Orchestrates APScheduler background jobs for scheduled market-close updates."""

    def __init__(self, config_path: str = "config/config.yaml", logger: Optional[logging.Logger] = None) -> None:
        self.config_path = config_path
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        cfg = AppConfig.from_yaml(self.config_path)
        cron_expr = cfg.week10.scheduler_cron
        tz = cfg.week10.timezone

        self.scheduler.add_job(
            func=run_full_pipeline_task,
            trigger=CronTrigger.from_crontab(cron_expr, timezone=tz),
            args=[self.config_path],
            id="daily_market_close_job",
            replace_existing=True,
        )
        self.scheduler.start()
        self.logger.info("PipelineScheduler started with cron: '%s' (%s)", cron_expr, tz)

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            self.logger.info("PipelineScheduler shut down successfully.")
