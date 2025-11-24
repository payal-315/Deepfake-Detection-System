from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from bson import ObjectId
import os

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://127.0.0.1:27017/?directConnection=true")
DATABASE_NAME = "deepfake_detection"

def serialize_doc(doc):
    if not doc:
        return doc
    doc = dict(doc)
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            doc[k] = str(v)
    return doc
# Async client for FastAPI
async def get_database():
    try:

        client = AsyncIOMotorClient(MONGODB_URL)
        return client[DATABASE_NAME]
    except Exception as e:
        print(e)

# Sync client for direct operations
def get_sync_database():
    client = MongoClient(MONGODB_URL)
    return client[DATABASE_NAME]

from pydantic_core import core_schema
from typing import Any
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(ObjectId),  # accept real ObjectId directly
                core_schema.no_info_after_validator_function(
                    cls.validate, core_schema.str_schema()
                ),  # accept string, convert to ObjectId
            ],
            serialization=core_schema.to_string_ser_schema(),  # when dumping → str(ObjectId)
        )
    
    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler):
        # JSON schema output (so OpenAPI/docs shows string type)
        return {"type": "string"}

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    username: str 
    email: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
    }

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class DetectionHistoryModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    filename: str
    file_type: str
    is_deepfake: bool
    confidence: float
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[dict] = None
    file_path: Optional[str] = None

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
    }

class AudioReferenceModel(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    reference_filename: Optional[str] = None
    test_filename: str
    similarity: Optional[float] = None
    verdict: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reference_path: Optional[str] = None
    test_path: Optional[str] = None

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
    }
