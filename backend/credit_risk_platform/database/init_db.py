from credit_risk_platform.database import models  # noqa: F401
from credit_risk_platform.database.session import Base, engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
