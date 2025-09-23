import os#check_vosk_model.py code

def check_vosk_model_dir(model_path):
    print(f"Checking Vosk model folder: {model_path}")
    if not os.path.exists(model_path):
        print("Model folder does not exist.")
        return False
    
    # List files and folders in the model path
    contents = os.listdir(model_path)
    print(f"Contents: {contents}")
    
    # Check for required folders/files
    required_folders = ['am', 'conf', 'graph', 'rescore']
    missing = [f for f in required_folders if f not in contents]
    if missing:
        print(f"Missing required files/folders: {missing}")
        return False
    
    print("Model folder looks good!")
    return True

# Put your actual model folder path here:
model_folder = r"models/vosk-model-small-en-us-0.15"

if __name__ == "__main__":
    valid = check_vosk_model_dir(model_folder)
    if not valid:
        print("Please ensure you have extracted the Vosk model properly and that your path is correct.")
    else:
        print("Vosk model setup is correct.")
