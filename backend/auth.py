
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# Static default user
DEFAULT_USER = {
    "id": "dev-user-001",
    "username": "developer",
    "email": "dev@example.com",
    "created_at": datetime.utcnow().isoformat()
}

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 36000

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# -------------------------
# ALWAYS RETURN DEFAULT USER
# -------------------------
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        # ⛔ DO NOT VERIFY TOKEN — accept any token
    token = credentials.credentials

    # If token exists, allow it
    if token:
        return DEFAULT_USER  

    raise HTTPException(
        status_code=401,
        detail="Missing authentication",
        headers={"WWW-Authenticate": "Bearer"},
    )

# -------------------------
# LOGIN (no DB, no password check)
# -------------------------
async def authenticate_user(username: str, password: str):
    # Always authenticate successfully
    return DEFAULT_USER

# -------------------------
# REGISTER (no DB)
# -------------------------
async def create_user(user_data):
    # Always "register" successfully
    return DEFAULT_USER



# from datetime import datetime, timedelta
# from typing import Optional
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from fastapi import HTTPException, status, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from database import UserModel, UserCreate, UserLogin, get_database, PyObjectId
# import os

# # Security configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 36000

# # Password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # JWT token security
# security = HTTPBearer()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     try :
#         return pwd_context.hash(password)
#     except Exception as e:
#         print(e)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# def verify_token(token: str) -> Optional[dict]:
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             return None
#         return payload
#     except JWTError:
#         return None

# async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserModel:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
    
#     token = credentials.credentials
#     payload = verify_token(token)
#     if payload is None:
#         raise credentials_exception
    
#     username: str = payload.get("sub")
#     if username is None:
#         raise credentials_exception
    
#     # Get user from database
#     db = await get_database()
#     user_data = await db.users.find_one({"username": username})
#     if user_data is None:
#         raise credentials_exception
    
#     return UserModel(**user_data)

# async def authenticate_user(username: str, password: str) -> Optional[UserModel]:
#     db = await get_database()
  
#     user_data = await db.users.find_one({"username": username})
 
#     if not user_data:
#         return None
 
#     user = UserModel(**user_data)

#     if not verify_password(password, user.hashed_password):
#         return None

#     return user

# async def create_user(user_data: UserCreate) -> UserModel:
#     db = await get_database()
    
#     # Check if user already exists
#     existing_user = await db.users.find_one({
#         "$or": [
#             {"username": user_data.username},
#             {"email": user_data.email}
#         ]
#     })
#     if existing_user != None:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username or email already registered"
#         )
  
#     # Create new user
#     hashed_password = get_password_hash(user_data.password)

#     user = UserModel(
#         username=user_data.username,
#         email=user_data.email,
#         hashed_password=hashed_password
#     )
#     result = await db.users.insert_one(user.dict(by_alias=True))

#     user.id = result.inserted_id
    
#     return user
