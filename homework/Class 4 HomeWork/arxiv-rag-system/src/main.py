import logging
from services.text_extraction_service import extract_text_from_folder

def main():
    
    folder_path = "../data/pdfs"
    max_files = 3
    
    pdf_contents = extract_text_from_folder(folder_path, max_files)

    for pdf_content in pdf_contents:
        print("*************************************")
        print(f"pdf content: \n{pdf_content}")

if __name__ == "__main__":
    main()
    