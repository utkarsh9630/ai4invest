#!/usr/bin/env python3
"""
Database Initialization Script
Run this script to create the database tables
"""

from app import app, db

def init_db():
    """Initialize the database with all tables"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print(" Database tables created successfully!")
        print("   Tables: User, Profile")
        
if __name__ == "__main__":
    init_db()