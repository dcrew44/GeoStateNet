"""
Setup script for running the state classifier in Colab.
Run this at the beginning of your Colab notebook.
"""
import os
import sys
from pathlib import Path
import subprocess


def setup_environment(
        github_repo="yourusername/state-classifier",
        branch="main",
        data_drive_path="/content/drive/MyDrive/50States10K",
        mount_drive=True
):
    """
    Set up the Colab environment for the state classifier.

    Args:
        github_repo (str): GitHub repository path
        branch (str): Branch to use
        data_drive_path (str): Path to the data in Google Drive
        mount_drive (bool): Whether to mount Google Drive
    """
    # Mount Google Drive if requested
    if mount_drive:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted.")

    # Clone the repository
    repo_url = f"https://github.com/{github_repo}.git"
    repo_dir = github_repo.split("/")[1]

    if os.path.exists(repo_dir):
        print(f"Repository directory {repo_dir} already exists. Pulling latest changes...")
        os.chdir(repo_dir)
        subprocess.run(["git", "pull", "origin", branch])
        os.chdir("..")
    else:
        print(f"Cloning repository from {repo_url}...")
        subprocess.run(["git", "clone", "-b", branch, repo_url])

    # Install the package
    print("Installing the package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", repo_dir])

    # Set up data symlinks or environment variables
    if os.path.exists(data_drive_path):
        print(f"Data found at {data_drive_path}")
        # Create environment variable to point to data
        os.environ["STATE_CLASSIFIER_DATA"] = data_drive_path
    else:
        print(f"Warning: Data path {data_drive_path} not found!")

    # Add the repository to Python path
    repo_path = os.path.abspath(repo_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    print(f"Setup complete! The codebase is available in /{repo_dir}/")
    return repo_path


# This allows importing from the setup module but also running it directly
if __name__ == "__main__":
    setup_environment()