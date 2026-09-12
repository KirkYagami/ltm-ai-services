from sqlalchemy import Column, Integer, String

from database import Base


class InventoryService(Base):
    __tablename__ = "inventory_services"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    service_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    service_secret_hash = Column(
        String(255),
        nullable=False
    )

    service_name = Column(
        String(200),
        nullable=False
    )
