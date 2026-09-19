"""
outbox_worker.py — Standalone process entrypoint for the OutboxRunner.
Can be executed natively via:
    python -m app.workers.outbox_worker
"""

import argparse
import asyncio
import logging
import signal
import sys

from app.db.session import async_session_maker
from app.services.outbox_service import OutboxRunner, outbox_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("outbox_worker_process")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Receptionist Outbox Worker Runner")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds when queue is empty (default: 2.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Maximum number of events claimed per batch (default: 20)",
    )
    parser.add_argument(
        "--stale-timeout",
        type=int,
        default=300,
        help="Timeout in seconds before reclaiming stale PROCESSING events (default: 300)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    runner = OutboxRunner(
        session_maker=async_session_maker,
        worker=outbox_worker,
        poll_interval_seconds=args.poll_interval,
        batch_size=args.batch_size,
        stale_timeout_seconds=args.stale_timeout,
    )

    loop = asyncio.get_running_loop()

    # Graceful shutdown handler
    def handle_shutdown(sig_name: str) -> None:
        logger.info("Received termination signal %s. Shutting down gracefully...", sig_name)
        runner.stop()

    # Register OS signal handlers if supported (e.g. Unix / Windows Proactor)
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s.name))

    logger.info("Starting Outbox Worker process...")
    try:
        await runner.run_continuous()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Outbox worker interrupted by user.")
        runner.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process exited cleanly.")
