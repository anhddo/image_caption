def get_image_names(file_path):
    file_names = []
    with open(file_path, 'r') as f:
        file_names = [line[: -1] for line in f.readlines()]

    return file_names
