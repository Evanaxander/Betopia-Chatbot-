import os
import shutil

def reset_project():
    project_root = os.getcwd()
    print(f"🧹 Resetting project at: {project_root}")

    # 1. Folders to ensure exist
    folders = ['rag', 'voice', 'app', 'data']

    for folder in folders:
        folder_path = os.path.join(project_root, folder)
        
        # Create folder if missing
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"📁 Created missing folder: {folder}")

        # 2. Force recreate __init__.py
        init_file = os.path.join(folder_path, "__init__.py")
        if os.path.exists(init_file):
            os.remove(init_file)
        
        with open(init_file, "w") as f:
            pass # Create empty file
        print(f"✅ Recreated: {folder}/__init__.py")

        # 3. Clear Python Cache (__pycache__)
        pycache = os.path.join(folder_path, "__pycache__")
        if os.path.exists(pycache):
            shutil.rmtree(pycache)
            print(f"🔥 Cleared cache for: {folder}")

    print("\n🚀 Project structure refreshed. Try running: python -m app.main")

if __name__ == "__main__":
    reset_project()