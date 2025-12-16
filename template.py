from pathlib import Path

def create_empty_files():
    """Creates an empty FastAPI project structure with blank files."""

    # List of files to create
    files_to_create = [
        "app.py",
        "Streamlit.py",
        "requirements.txt",
        "templates/home.html",
        "templates/profit.html",
        "templates/sales.html",
        "static/style.css",
        "static/profit.css",
        "static/sales.css"
    ]

    print("Starting file structure creation...")

    # Create directories if they don’t exist
    Path("templates").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    print("Created 'templates/' and 'static/' directories.")

    # Create empty files
    for file_path in files_to_create:
        full_path = Path(file_path)
        with open(full_path, "w", encoding="utf-8") as f:
            pass  # create empty file
        print(f"✅ Created empty file: {file_path}")

    print("\n✅ All files created successfully!")
    print("----------------------------------------------------------")
    print("You can now add code to these files and build your FastAPI app.")
    print("----------------------------------------------------------")

if __name__ == "__main__":
    create_empty_files()
