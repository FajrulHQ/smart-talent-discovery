from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, utility
from .colpali_retriever import MilvusColpali

load_dotenv()

GD_FOLDER_ID = os.getenv("GD_FOLDER_ID")  # Replace with actual folder ID
LOCAL_FOLDER = "./data"       # Change to desired local path

class Configuration:
    """
    Base class for Google Drive authentication.
    and Milvus configuration
    """
    
    def __init__(self, milvus_uri: str, collection_name: str):
        # self.authenticate()
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.client = MilvusClient(uri=milvus_uri)
        self.milvus = MilvusColpali(self.client, self.collection_name)
        self.milvus()

    def authenticate(self):
        """Authenticate with Google Drive."""
        self.gauth = GoogleAuth()
        self.gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(self.gauth)

class GD2MilvusManager(Configuration):
    """
    Class to list and download files (including subfolders) from Google Drive
    Store embedded document with colpali to Milvus.
    """

    def list_files_and_folders(self, folder_id):
        """List all files and folders inside a given folder."""
        query = f"'{folder_id}' in parents and trashed=false"
        items = self.drive.ListFile({'q': query}).GetList()

        files = []
        folders = []

        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                folders.append((item['title'], item['id']))  # It's a folder
            else:
                files.append((item['title'], item['id']))  # It's a file
        
        return files, folders
    
    def download_files_recursive(self, folder_id, local_folder):
        """Recursively download all files from a Google Drive folder and its subfolders."""
        os.makedirs(local_folder, exist_ok=True)  # Ensure folder exists

        files, folders = self.list_files_and_folders(folder_id)

        if len(files) == 0 and \
            ('portfolio' in local_folder.lower() or 
             'resume' in local_folder.lower()
        ):
            print("No file in this folder")
            
        # Download files in the current folder
        for file_name, file_id in files:
            print(f"{file_name} ✅")
            if file_name in os.listdir(local_folder):
                print(f" ⦿ Already Downloaded ✅")
                continue
            file = self.drive.CreateFile({'id': file_id})
            file.GetContentFile(os.path.join(local_folder, file_name))
            print(f" ⦿ Downloaded ✅")

        # Recursively download from subfolders
        for folder_name, subfolder_id in folders:
            print(f"{folder_name} 📁")
            subfolder_path = os.path.join(local_folder, folder_name)
            self.download_files_recursive(subfolder_id, subfolder_path)

if __name__ == "__main__":
    gdm = GD2MilvusManager()
    gdm.download_files_recursive(GD_FOLDER_ID, LOCAL_FOLDER)