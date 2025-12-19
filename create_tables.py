import os
from app import app, db

# This script creates all database tables
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(" Database tables created successfully!")
        print("Tables: users, profiles")