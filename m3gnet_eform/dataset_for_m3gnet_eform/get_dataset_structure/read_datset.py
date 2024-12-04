import json
import json
import os

def analyze_dataset_structure(filename):
    """
    Analyze the structure of labels in the dataset
    Print the labels from the first entry and verify if all entries follow the same structure
    """
    print(f"Reading file: {filename}")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Data is not a list")
            return
            
        print(f"Total number of materials in dataset: {len(data)}")
        
        # Get labels from first entry
        if len(data) == 0:
            print("Dataset is empty")
            return
            
        first_entry = data[0]
        if not isinstance(first_entry, dict):
            print(f"Unexpected data type for entry: {type(first_entry)}")
            return
            
        print("\nLabels in dataset:")
        print("=" * 50)
        for key in first_entry.keys():
            print(f"- {key}")
            
        # Verify if all entries have the same labels
        print("\nVerifying label consistency across dataset...")
        base_keys = set(first_entry.keys())
        
        for idx, entry in enumerate(data[1:], 1):
            current_keys = set(entry.keys())
            if current_keys != base_keys:
                print(f"\nMismatch found at entry {idx}:")
                print(f"Missing keys: {base_keys - current_keys}")
                print(f"Extra keys: {current_keys - base_keys}")
                
        print("\nVerification complete.")
        print(f"First entry example values for each label:")
        print("=" * 50)
        for key, value in first_entry.items():
            value_type = type(value).__name__
            if isinstance(value, (list, dict)):
                size = len(value)
                print(f"{key}: Type={value_type}, Size={size}")
            else:
                print(f"{key}: Type={value_type}, Value={value}")
    except Exception as e:
        raise

# Run analysis
try:
    analyze_dataset_structure("mp.2018.6.1.json")
except FileNotFoundError:
    print("Error: File not found")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format - {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    
def analyze_dataset_details(filename):
    """
    Analyze the dataset with special attention to different structure types
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Get standard structure from first entry
    standard_keys = set(data[0].keys())
    print(f"Standard labels ({len(standard_keys)}):")
    for key in sorted(standard_keys):
        print(f"- {key}")
    
    # Find all unique structure types
    structure_types = {}
    for idx, entry in enumerate(data):
        key_set = frozenset(entry.keys())
        if key_set not in structure_types:
            structure_types[key_set] = {
                'count': 0,
                'example_idx': idx,
                'keys': set(entry.keys())
            }
        structure_types[key_set]['count'] += 1
    
    print(f"\nFound {len(structure_types)} different structure types:")
    for i, (key_set, info) in enumerate(structure_types.items(), 1):
        diff_keys = info['keys'] - standard_keys
        missing_keys = standard_keys - info['keys']
        print(f"\nStructure Type {i}:")
        print(f"Count: {info['count']} entries")
        print(f"Example Index: {info['example_idx']}")
        if diff_keys:
            print(f"Additional keys: {diff_keys}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
            
        # Print example values for different keys
        example_entry = data[info['example_idx']]
        if diff_keys:
            print("\nExample values for additional keys:")
            for key in diff_keys:
                print(f"{key}: {example_entry[key]}")

# Run analysis
try:
    analyze_dataset_details("mp.2018.6.1.json")
except FileNotFoundError:
    print("Error: File not found")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format - {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")