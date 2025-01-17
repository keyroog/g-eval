#count the record of a file json
import json
import os

def count_record(file):
    """
    Count the record of a file json
    """
    with open(file, "r") as f:
        data = json.load(f)
    return len(data["contexts"])

#count the record of a file json

def main():
    """
    Main function
    """
    print(count_record("datasets/dstc9_data.json"))

if __name__ == "__main__":
    main()