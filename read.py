import pandas as pd
def read_data_fox(filename):
    df = pd.read_csv(filename,
                   sep=',', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot(index='user', columns='item', values='rate')
    matrix.fillna(0, inplace=True)
    matrix_list = matrix.iloc[:,:].values

    return matrix_list

def read_data_ori(filename):
    df = pd.read_csv(filename,
                   sep='\t', names=['user','item','rate','time'], engine='python', encoding='latin-1')
    matrix = df.pivot(index='user', columns='item', values='rate')
    matrix.fillna(0, inplace=True)
    matrix_list = matrix.iloc[:,:].values

    return matrix_list

path = "result/mon1.csv"
data = read_data_fox(path)
print(data.shape)
path_ = 'result/ori.csv'
dataset = read_data_ori(path_)
print(dataset.shape)

# dataset = pd.read_csv('ml-100k/u.data', delimiter='\t', header=None, names=['userID', 'movieID', 'rating', 'timestemp'])

# data_ = dataset.to_numpy()
# print(data_)
