"""add appointments, tenant isolation, and deferred outbox retries

Revision ID: 20260918_0003
Revises: 20260917_0002
Create Date: 2026-09-18 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260918_0003"
down_revision = "20260917_0002"
branch_labels = None
depends_on = None


def _index_names(inspector: sa.Inspector, table_name: str) -> set[str]:
    return {index["name"] for index in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())

    # The original initial migration accidentally omitted this ORM table.  Some
    # development databases have it only because the application used
    # metadata.create_all(), so this migration supports both states.
    if "appointments" not in table_names:
        op.create_table(
            "appointments",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("tenant_id", sa.String(length=50), nullable=False, server_default="default"),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("customer_name", sa.String(length=100), nullable=False),
            sa.Column("customer_phone", sa.String(length=30), nullable=False),
            sa.Column("customer_email", sa.String(length=255), nullable=True),
            sa.Column("service", sa.String(length=100), nullable=False, server_default="General Consultation"),
            sa.Column("day", sa.String(length=20), nullable=False),
            sa.Column("time_str", sa.String(length=20), nullable=False),
            sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="confirmed"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        op.create_index("ix_appointments_id", "appointments", ["id"], unique=False)
        op.create_index("ix_appointments_tenant_id", "appointments", ["tenant_id"], unique=False)
        op.create_index("ix_appointments_user_id", "appointments", ["user_id"], unique=False)
        op.create_index("ix_appointments_customer_phone", "appointments", ["customer_phone"], unique=False)
    else:
        columns = {column["name"] for column in inspector.get_columns("appointments")}
        if "tenant_id" not in columns:
            op.add_column(
                "appointments",
                sa.Column("tenant_id", sa.String(length=50), nullable=False, server_default="default"),
            )

        inspector = sa.inspect(bind)
        existing_indexes = _index_names(inspector, "appointments")
        if "ix_appointments_tenant_id" not in existing_indexes:
            op.create_index("ix_appointments_tenant_id", "appointments", ["tenant_id"], unique=False)

    inspector = sa.inspect(bind)
    if "uq_appointments_active_tenant_slot" not in _index_names(inspector, "appointments"):
        # A partial unique index is portable across the supported SQLite and
        # PostgreSQL dialects.  It prevents concurrent active reservations
        # while allowing a cancelled slot to be rebooked.
        op.create_index(
            "uq_appointments_active_tenant_slot",
            "appointments",
            ["tenant_id", "day", "time_str"],
            unique=True,
            postgresql_where=sa.text("status <> 'cancelled'"),
            sqlite_where=sa.text("status <> 'cancelled'"),
        )

    inspector = sa.inspect(bind)
    outbox_columns = {column["name"] for column in inspector.get_columns("outbox_events")}
    if "next_attempt_at" not in outbox_columns:
        op.add_column("outbox_events", sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True))
        op.execute(sa.text("UPDATE outbox_events SET next_attempt_at = created_at WHERE next_attempt_at IS NULL"))

    inspector = sa.inspect(bind)
    outbox_columns = {column["name"] for column in inspector.get_columns("outbox_events")}
    if "claim_token" not in outbox_columns:
        op.add_column("outbox_events", sa.Column("claim_token", sa.String(length=36), nullable=True))

    inspector = sa.inspect(bind)
    if "ix_outbox_events_next_attempt_at" not in _index_names(inspector, "outbox_events"):
        op.create_index("ix_outbox_events_next_attempt_at", "outbox_events", ["next_attempt_at"], unique=False)
    if "ix_outbox_events_claim_token" not in _index_names(inspector, "outbox_events"):
        op.create_index("ix_outbox_events_claim_token", "outbox_events", ["claim_token"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "ix_outbox_events_next_attempt_at" in _index_names(inspector, "outbox_events"):
        op.drop_index("ix_outbox_events_next_attempt_at", table_name="outbox_events")
    if "ix_outbox_events_claim_token" in _index_names(inspector, "outbox_events"):
        op.drop_index("ix_outbox_events_claim_token", table_name="outbox_events")
    if "next_attempt_at" in {column["name"] for column in inspector.get_columns("outbox_events")}:
        op.drop_column("outbox_events", "next_attempt_at")
    if "claim_token" in {column["name"] for column in inspector.get_columns("outbox_events")}:
        op.drop_column("outbox_events", "claim_token")

    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "appointments" in table_names:
        existing_indexes = _index_names(inspector, "appointments")
        if "uq_appointments_active_tenant_slot" in existing_indexes:
            op.drop_index("uq_appointments_active_tenant_slot", table_name="appointments")
        if "ix_appointments_tenant_id" in existing_indexes:
            op.drop_index("ix_appointments_tenant_id", table_name="appointments")
        # To prevent accidental loss of appointment history during migrations rollback,
        # the table is preserved rather than blindly dropped.
