import os


PATH_TO_TRAIN = 'data/train'
PATH_TO_TEST = 'data/test'
PATH_TO_VALIDATION = 'data/validation'


def generate_dictionary(path):
    df = dict()
    for category in os.listdir(path):
        path_to_files = os.path.join(path, category)
        for file in os.listdir(path_to_files):
            df[file] = category
    return df


with open('data/train.csv', 'w') as file:
    df = generate_dictionary(PATH_TO_TRAIN)
    for key in df.keys():
        s = os.path.join(PATH_TO_TRAIN, df[key], key) + ',' + df[key] + '\n'
        file.write(s)

with open('data/validation.csv', 'w') as file:
    df = generate_dictionary(PATH_TO_VALIDATION)
    for key in df.keys():
        s = os.path.join(PATH_TO_VALIDATION, df[key], key) + ',' + df[key] + '\n'
        file.write(s)

with open('data/test.csv', 'w') as file:
    df = generate_dictionary(PATH_TO_TEST)
    for key in df.keys():
        s = os.path.join(PATH_TO_TEST, df[key], key) + ',' + df[key] + '\n'
        file.write(s)
