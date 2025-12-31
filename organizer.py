import os 
import shutil

base_folder = "test folder"

files= os.listdir(base_folder)

for file in files:
    file_path = os.path.join(base_folder,file)

    if os.path.isfile(file_path):
        name,ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in ['.jpeg','.jpg','.png']:
            folder = "Images"
        elif ext == '.pdf':
            folder = "PDFs"
        elif ext in ['.docx','.xlsx','.csv']:
            folder = "Documents"
        elif ext == '.txt':
            folder = "TextFile"
        elif ext =='.exe':
            folder = "Applications"
        elif ext =='.css':
            folder = "CSS files"
        elif ext =='.html':
            folder ="HTML files"
        elif ext =='.py':
            folder = "Python files"
        else:
            folder ="Others"
    
    destination_folder=os.path.join(base_folder,folder)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    shutil.move(file_path,os.path.join(destination_folder,file))
print("Files organized successfully!")