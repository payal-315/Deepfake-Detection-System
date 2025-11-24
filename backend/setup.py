#!/usr/bin/env python3
"""
Setup script for Deepfake Detection API
This script initializes the MongoDB database and creates necessary indexes.
"""

import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import OperationFailure

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://127.0.0.1:27017/?directConnection=true")

DATABASE_NAME = "deepfake_detection"

async def setup_database():
    """Initialize the database and create indexes"""
    print("🔧 Setting up Deepfake Detection Database...")
    
    try:
        # Test connection
        client = AsyncIOMotorClient(MONGODB_URL)
        await client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        db = client[DATABASE_NAME]
        
        # Create collections if they don't exist
        collections = ['users', 'detection_history', 'audio_references']
        
        for collection_name in collections:
            try:
                await db.create_collection(collection_name)
                print(f"✅ Created collection: {collection_name}")
            except OperationFailure as e:
                if "already exists" in str(e):
                    print(f"ℹ️  Collection already exists: {collection_name}")
                else:
                    print(f"⚠️  Warning creating collection {collection_name}: {e}")
        
        # Create indexes
        print("\n📊 Creating database indexes...")
        
        # Users collection indexes
        try:
            await db.users.create_index("username", unique=True)
            print("✅ Created unique index on users.username")
        except Exception as e:
            print(f"⚠️  Warning creating username index: {e}")
            
        try:
            await db.users.create_index("email", unique=True)
            print("✅ Created unique index on users.email")
        except Exception as e:
            print(f"⚠️  Warning creating email index: {e}")
        
        # Detection history indexes
        try:
            await db.detection_history.create_index("user_id")
            print("✅ Created index on detection_history.user_id")
        except Exception as e:
            print(f"⚠️  Warning creating user_id index: {e}")
            
        try:
            await db.detection_history.create_index([("timestamp", -1)])
            print("✅ Created index on detection_history.timestamp")
        except Exception as e:
            print(f"⚠️  Warning creating timestamp index: {e}")
        
        # Audio references indexes
        try:
            await db.audio_references.create_index("user_id")
            print("✅ Created index on audio_references.user_id")
        except Exception as e:
            print(f"⚠️  Warning creating user_id index: {e}")
            
        try:
            await db.audio_references.create_index([("timestamp", -1)])
            print("✅ Created index on audio_references.timestamp")
        except Exception as e:
            print(f"⚠️  Warning creating timestamp index: {e}")
        
        print("\n🎉 Database setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Install Python dependencies: pip install -r requirements.txt")
        print("2. Set environment variables (optional):")
        print("   - MONGODB_URL: Your MongoDB connection string")
        print("   - SECRET_KEY: A secure secret key for JWT tokens")
        print("3. Start the backend server: python main.py")
        print("4. Start the frontend: cd ../frontend && npm run dev")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("1. Make sure MongoDB is running")
        print("2. Check your MONGODB_URL environment variable")
        print("3. Ensure you have network access to MongoDB")
        return False
    
    finally:
        client.close()
    
    return True

def setup_sync():
    """Synchronous version for direct execution"""
    try:
        client = MongoClient(MONGODB_URL)
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        db = client[DATABASE_NAME]
        
        # Create collections
        collections = ['users', 'detection_history', 'audio_references']
        
        for collection_name in collections:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
                print(f"✅ Created collection: {collection_name}")
            else:
                print(f"ℹ️  Collection already exists: {collection_name}")
        
        # Create indexes
        print("\n📊 Creating database indexes...")
        
        # Users indexes
        try:
            db.users.create_index("username", unique=True)
            print("✅ Created unique index on users.username")
        except Exception as e:
            print(f"⚠️  Warning creating username index: {e}")
            
        try:
            db.users.create_index("email", unique=True)
            print("✅ Created unique index on users.email")
        except Exception as e:
            print(f"⚠️  Warning creating email index: {e}")
        
        # Detection history indexes
        try:
            db.detection_history.create_index("user_id")
            print("✅ Created index on detection_history.user_id")
        except Exception as e:
            print(f"⚠️  Warning creating user_id index: {e}")
            
        try:
            db.detection_history.create_index([("timestamp", -1)])
            print("✅ Created index on detection_history.timestamp")
        except Exception as e:
            print(f"⚠️  Warning creating timestamp index: {e}")
        
        # Audio references indexes
        try:
            db.audio_references.create_index("user_id")
            print("✅ Created index on audio_references.user_id")
        except Exception as e:
            print(f"⚠️  Warning creating user_id index: {e}")
            
        try:
            db.audio_references.create_index([("timestamp", -1)])
            print("✅ Created index on audio_references.timestamp")
        except Exception as e:
            print(f"⚠️  Warning creating timestamp index: {e}")
        
        print("\n🎉 Database setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Install Python dependencies: pip install -r requirements.txt")
        print("2. Set environment variables (optional):")
        print("   - MONGODB_URL: Your MongoDB connection string")
        print("   - SECRET_KEY: A secure secret key for JWT tokens")
        print("3. Start the backend server: python main.py")
        print("4. Start the frontend: cd ../frontend && npm run dev")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("1. Make sure MongoDB is running")
        print("2. Check your MONGODB_URL environment variable")
        print("3. Ensure you have network access to MongoDB")
        return False
    
    finally:
        client.close()
    
    return True

if __name__ == "__main__":
    print("🚀 Deepfake Detection Database Setup")
    print("=" * 50)
    
    # Try async first, fallback to sync
    try:
        asyncio.run(setup_database())
    except Exception:
        print("Async setup failed, trying sync setup...")
        setup_sync()
