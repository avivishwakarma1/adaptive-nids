
from datetime import datetime, timezone
from getpass import getpass

from pymongo.errors import PyMongoError

from app.config import get_settings
from app.db import create_mongo_client, ensure_indexes
from app.security import hash_password


def main():
    settings = get_settings()
    client = create_mongo_client(settings.mongodb_uri)

    if client is None:
        print("ERROR: MongoDB URI is not configured.")
        return

    try:
        client.admin.command("ping")
        db = client[settings.mongodb_database]
        ensure_indexes(db)

        email = input("Admin email: ").strip().lower()
        display_name = input("Admin display name: ").strip()
        password = getpass("Admin password (hidden): ")
        confirmation = getpass("Confirm password: ")

        if not email or "@" not in email:
            print("ERROR: Enter a valid email address.")
            return

        if not display_name:
            print("ERROR: Display name is required.")
            return

        if len(password) < 12:
            print("ERROR: Use a password with at least 12 characters.")
            return

        if password != confirmation:
            print("ERROR: Passwords do not match.")
            return

        if db.users.find_one({"email_normalized": email}):
            print("ERROR: An account with this email already exists.")
            return

        user = {
            "email": email,
            "email_normalized": email,
            "display_name": display_name,
            "password_hash": hash_password(password),
            "role": "admin",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "last_login_at": None,
        }

        db.users.insert_one(user)
        print("SUCCESS: Admin account created.")
        print("Email:", email)
        print("Role: admin")

    except PyMongoError as exc:
        print("Database error:", type(exc).__name__)
        print("Check your database connection and indexes.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
