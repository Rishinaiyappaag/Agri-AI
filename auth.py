"""
auth.py - Authentication Module
Handles user registration, login, and password hashing
"""

import bcrypt
from datetime import datetime

def hash_password(password):
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt(rounds=10)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password, hashed_password):
    """Verify password against hashed password"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_user(mongo, full_name, email, password):
    """
    Create new user in MongoDB
    Returns: (success, message, user_id)
    """
    try:
        # Check if user already exists
        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user:
            return False, "❌ Email already registered!", None
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Create user document
        user_data = {
            "full_name": full_name,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "last_login": None,
            "is_active": True
        }
        
        # Insert into MongoDB
        result = mongo.db.users.insert_one(user_data)
        
        return True, "✅ Account created successfully! Please login.", str(result.inserted_id)
    
    except Exception as e:
        print(f"Error creating user: {str(e)}")
        return False, f"❌ Error: {str(e)}", None

def verify_login(mongo, email, password):
    """
    Verify user login credentials
    Returns: (success, message, user_data)
    """
    try:
        # Find user by email
        user = mongo.db.users.find_one({"email": email})
        
        if not user:
            return False, "❌ Email not found!", None
        
        # Verify password
        if not verify_password(password, user["password"]):
            return False, "❌ Invalid password!", None
        
        # Update last login
        mongo.db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        return True, "✅ Login successful!", user
    
    except Exception as e:
        print(f"Error during login: {str(e)}")
        return False, f"❌ Error: {str(e)}", None

def get_user_by_email(mongo, email):
    """Get user document by email"""
    return mongo.db.users.find_one({"email": email})

def get_user_by_id(mongo, user_id):
    """Get user document by ID"""
    from bson.objectid import ObjectId
    return mongo.db.users.find_one({"_id": ObjectId(user_id)})