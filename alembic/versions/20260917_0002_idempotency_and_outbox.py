"""add idempotency_records and outbox_events tables

Revision ID: 20260917_0002
Revises: 20260607_0001
Create Date: 2026-09-17 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260917_0002"
down_revision = "20260607_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. idempotency_records table
    op.create_table(
        "idempotency_records",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(length=50), nullable=False, server_default="default"),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("operation", sa.String(length=50), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="COMPLETED"),
        sa.Column("result_data", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("tenant_id", "key", "operation", name="uq_idempotency_tenant_key_operation"),
    )
    op.create_index(op.f("ix_idempotency_records_id"), "idempotency_records", ["id"], unique=False)
    op.create_index(op.f("ix_idempotency_records_tenant_id"), "idempotency_records", ["tenant_id"], unique=False)
    op.create_index(op.f("ix_idempotency_records_key"), "idempotency_records", ["key"], unique=False)
    op.create_index(op.f("ix_idempotency_records_operation"), "idempotency_records", ["operation"], unique=False)

    # 2. outbox_events table
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(op.f("ix_outbox_events_id"), "outbox_events", ["id"], unique=False)
    op.create_index(op.f("ix_outbox_events_event_type"), "outbox_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_outbox_events_status"), "outbox_events", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_outbox_events_status"), table_name="outbox_events")
    op.drop_index(op.f("ix_outbox_events_event_type"), table_name="outbox_events")
    op.drop_index(op.f("ix_outbox_events_id"), table_name="outbox_events")
    op.drop_table("outbox_events")

    op.drop_index(op.f("ix_idempotency_records_operation"), table_name="idempotency_records")
    op.drop_index(op.f("ix_idempotency_records_key"), table_name="idempotency_records")
    op.drop_index(op.f("ix_idempotency_records_tenant_id"), table_name="idempotency_records")
    op.drop_index(op.f("ix_idempotency_records_id"), table_name="idempotency_records")
    op.drop_table("idempotency_records")
