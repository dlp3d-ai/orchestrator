import argparse
import asyncio

from pymongo import AsyncMongoClient


async def main(args):
    """Initialize MongoDB memory database and orchestrator user.

    This function connects to MongoDB as admin, creates the memory database if it
    doesn't exist, creates or updates the orchestrator user with appropriate
    permissions, and validates the user's access.

    Args:
        args (argparse.Namespace):
            Command line arguments containing MongoDB connection details and
            user credentials.
    """
    async with AsyncMongoClient(
        host=args.host, port=args.port, username=args.admin_username, password=args.admin_password, authSource="admin"
    ) as client:
        try:
            # Test connection
            await client.admin.command("ping")

            # Get target database
            db = client[args.memory_database]

            # Check if database exists
            db_exists = args.memory_database in await client.list_database_names()

            # Check if user already exists
            existing_users = await db.command("usersInfo")
            user_exists = any(user["user"] == args.orchestrator_username for user in existing_users["users"])

            # Check web database and user
            web_db = client[args.web_database]
            web_db_exists = args.web_database in await client.list_database_names()

            if web_db_exists:
                web_existing_users = await web_db.command("usersInfo")
                web_user_exists = any(
                    user["user"] == args.orchestrator_username for user in web_existing_users["users"]
                )
            else:
                web_user_exists = False

            # If both memory database and user exist, and web database user also exists, exit silently
            if db_exists and user_exists and web_user_exists:
                return

            # Create memory database if it doesn't exist
            if not db_exists:
                await db.create_collection("_init")
                print(f"Database '{args.memory_database}' created")

            # Create or update memory database user
            if user_exists:
                # Update user password
                await db.command("updateUser", args.orchestrator_username, pwd=args.orchestrator_password)
                print(f"User '{args.orchestrator_username}' password updated in memory database")
            else:
                # Create new user with full database permissions
                await db.command(
                    "createUser",
                    args.orchestrator_username,
                    pwd=args.orchestrator_password,
                    roles=[
                        {"role": "readWrite", "db": args.memory_database},
                        {"role": "dbAdmin", "db": args.memory_database},
                    ],
                )
                print(f"User '{args.orchestrator_username}' created successfully in memory database")

            # Handle web database
            if not web_user_exists:
                # Create web database if it doesn't exist
                if not web_db_exists:
                    await web_db.create_collection("_init")
                    print(f"Database '{args.web_database}' created")

                # Create user in web database with readWrite role
                await web_db.command(
                    "createUser",
                    args.orchestrator_username,
                    pwd=args.orchestrator_password,
                    roles=[
                        {"role": "readWrite", "db": args.web_database},
                    ],
                )
                print(f"User '{args.orchestrator_username}' created successfully in web database")

            # Verify user permissions
            print(f"User '{args.orchestrator_username}' has full permissions on database '{args.memory_database}'")
            print(f"User '{args.orchestrator_username}' has readWrite permissions on database '{args.web_database}'")
            print(f"Memory database auth source: {args.memory_database}")
            print(f"Web database auth source: {args.web_database}")
            print(
                f"Memory connection string: mongodb://{args.orchestrator_username}:{args.orchestrator_password}@{args.host}:{args.port}/{args.memory_database}?authSource={args.memory_database}"
            )
            print(
                f"Web connection string: mongodb://{args.orchestrator_username}:{args.orchestrator_password}@{args.host}:{args.port}/{args.web_database}?authSource={args.web_database}"
            )

            # Test new user connection
            await test_user_connection(args)

        except Exception as e:
            print(f"Operation failed: {e}")
            raise


async def test_user_connection(args):
    """Test connection and permissions for the newly created user.

    This function validates that the orchestrator user can successfully connect
    to MongoDB and perform read/write operations on both memory and web databases.

    Args:
        args (argparse.Namespace):
            Command line arguments containing MongoDB connection details and
            user credentials.
    """
    try:
        # Test memory database connection
        memory_client = AsyncMongoClient(
            host=args.host,
            port=args.port,
            username=args.orchestrator_username,
            password=args.orchestrator_password,
            authSource=args.memory_database,
        )

        # Test memory database connection
        await memory_client.admin.command("ping")
        print(f"User '{args.orchestrator_username}' memory database connection test successful")

        # Test memory database read/write permissions
        memory_db = memory_client[args.memory_database]
        memory_test_collection = memory_db["test_collection"]

        # Test write operation
        test_doc = {"test": "memory_data", "timestamp": "2024-01-01"}
        result = await memory_test_collection.insert_one(test_doc)
        print(f"Memory database write test successful, document ID: {result.inserted_id}")

        # Test read operation
        found_doc = await memory_test_collection.find_one({"_id": result.inserted_id})
        print(f"Memory database read test successful: {found_doc}")

        # Clean up memory test data
        await memory_test_collection.delete_one({"_id": result.inserted_id})
        print("Memory database test data cleaned up")

        await memory_client.close()

        # Test web database connection
        web_client = AsyncMongoClient(
            host=args.host,
            port=args.port,
            username=args.orchestrator_username,
            password=args.orchestrator_password,
            authSource=args.web_database,
        )

        # Test web database connection
        await web_client.admin.command("ping")
        print(f"User '{args.orchestrator_username}' web database connection test successful")

        # Test web database read/write permissions
        web_db = web_client[args.web_database]
        web_test_collection = web_db["test_collection"]

        # Test write operation
        test_doc = {"test": "web_data", "timestamp": "2024-01-01"}
        result = await web_test_collection.insert_one(test_doc)
        print(f"Web database write test successful, document ID: {result.inserted_id}")

        # Test read operation
        found_doc = await web_test_collection.find_one({"_id": result.inserted_id})
        print(f"Web database read test successful: {found_doc}")

        # Clean up web test data
        await web_test_collection.delete_one({"_id": result.inserted_id})
        print("Web database test data cleaned up")

        await web_client.close()
        print("✅ User permission verification completed for both databases!")

    except Exception as e:
        print(f"User connection test failed: {e}")
        raise


def parse_args():
    """Parse command line arguments for MongoDB setup.

    Returns:
        argparse.Namespace:
            Parsed command line arguments containing MongoDB connection details
            and user credentials.
    """
    parser = argparse.ArgumentParser(description="Initialize MongoDB memory database and orchestrator user")
    parser.add_argument("--admin_username", type=str, default="admin", help="MongoDB admin username for authentication")
    parser.add_argument("--admin_password", type=str, required=True, help="MongoDB admin password for authentication")
    parser.add_argument("--host", type=str, required=True, help="MongoDB server host address")
    parser.add_argument("--port", type=int, required=True, help="MongoDB server port number")
    parser.add_argument(
        "--web_database",
        type=str,
        default="web",
        help="Name of the webpage database to create (default: web)",
    )
    parser.add_argument(
        "--memory_database",
        type=str,
        default="character_memory",
        help="Name of the memory database to create (default: character_memory)",
    )
    parser.add_argument(
        "--orchestrator_username", type=str, required=True, help="Username for the orchestrator user to create"
    )
    parser.add_argument(
        "--orchestrator_password", type=str, required=True, help="Password for the orchestrator user to create"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
