
# load the rmb dataset into a dataframe data structure, where it has columns:

# 1. category
# 2. harmless/helpfulness
# 3. dataset_name
import json
import os

for split in ["BoN_set", "Pairwise_set"]:
    for hh_type in ["Harmlessness", "Helpfulness"]:
        if hh_type == "Harmlessness":
            # recursively read files in the directory. 
            pass 
        else:
            # list all dirs; 
            for dir_name in os.listdir(f"RMB_dataset/{split}/{hh_type}"):
                # Get the full path to the directory
                dir_path = os.path.join(f"RMB_dataset/{split}/{hh_type}", dir_name)
                # Check if it's a directory
                if os.path.isdir(dir_path):
                    # List to store all JSON data
                    json_data_list = []
                    
                    # List all JSON files in the directory
                    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
                    
                    # Process each JSON file
                    for json_file in json_files:
                        json_file_path = os.path.join(dir_path, json_file)
                        
                        # Read the JSON file into a dictionary
                        try:
                            with open(json_file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                json_data_list.append(data)
                        except Exception as e:
                            print(f"Error reading {json_file_path}: {e}")
                    
                    print(f"Loaded {len(json_data_list)} JSON files from {dir_path}")


