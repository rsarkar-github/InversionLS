import json


def write_json(filename, key, val):
    """
    Append a .json file

    :param filename: str
        Path of .json file to append
    :param key:
        Key of entry to add
    :param val:
        Value of entry to add

    :return:
    """
    with open(filename, 'r+') as file:

        file_data = json.load(file)
        file_data[key] = val
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()

def update_json(filename, key, val):
    """
    Update a .json file

    :param filename: str
        Path of .json file to update
    :param key:
        Key of entry to update
    :param val:
        Value of entry to update

    :return:
    """
    with open(filename, 'r+') as file:

        file_data = json.load(file)
        file_data[key] = val
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()
