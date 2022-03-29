import dill

def read_data(path):
    with open(path, 'rb') as in_strm:
        data = dill.load(in_strm)
    return data

def save_data(file,path):
    with open(path, "wb") as pkl_handle:
        dill.dump(file, pkl_handle)

def check_length(data):
    for name in data.keys():
        print(name+':######', len(data[name]))
