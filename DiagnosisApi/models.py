from sqlalchemy import Boolean, Column, ForeignKey, Numeric, Integer, String
from sqlalchemy.orm import relationship

from database import Base

class Client(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(String)
    confidenceLevel = Column(Numeric(10, 2))
