from read_preprocess import *

path = "result/mon1.csv"
data = read_data_coma(path)
print(data.shape)
path_ = 'result/ori.csv'
dataset = read_data_t(path_)
print(dataset.shape)
print(dataset[0, :])

np.savetxt('ml-100k_result/train.txt', dataset, fmt='%i', delimiter='\t')
