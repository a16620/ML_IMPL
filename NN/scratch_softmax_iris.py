import numpy as np
import pandas

class IntCategoryEncoder:
    def __init__(self):
        self._category_map = {}
    
    def fit(self, a:np.ndarray):
        category = np.unique(a)
        
        for i in range(category.shape[0]):
            self._category_map[category[i]] = i
    
    def transform(self, a: np.ndarray):
        def trn(x):
            return self._category_map.get(x)
        vfnTrn = np.vectorize(trn)
        return vfnTrn(a)

def normalize(x:np.ndarray): #표준화 Z점수
    mean = x.mean()
    std = np.std(x)
    vfnNorm = np.vectorize(lambda d: (d-mean)/std)
    normArr = vfnNorm(x)
    return normArr

ds = pandas.read_csv('iris.csv')

f1 = normalize(ds['sepal_length'].values)
f2 = normalize(ds['sepal_width'].values)

x0 = np.c_[f1, f2]
t0 = ds['species'].values

encoder = IntCategoryEncoder()
encoder.fit(t0)

t0 = encoder.transform(t0)

nCls = len(encoder._category_map)
t0 = np.eye(nCls)[t0]

def softmax(x: np.ndarray):
    x = x-np.max(x)
    x = np.exp(x)
    x = x/x.sum()
    return x

def loss(pred: np.ndarray, target: np.ndarray):
    return -(np.log(pred)*target).sum()

def grad_loss_weight(pred: np.ndarray, ft: np.ndarray, target: np.ndarray):
    return -((ft.transpose()*target)-(ft.transpose()*pred))

def grad_loss_bias(pred: np.ndarray, target: np.ndarray):
    return -(target-pred)

learning_rate = 0.01
feature_size = 2
class_count = nCls

W = np.random.uniform(-1, 1, size=(feature_size, class_count))
b = np.random.uniform(-1, 1, size=(1, class_count))

def fit(x, target):
    global W, b
    pred = eval(x)

    grad_w = grad_loss_weight(pred, x, target)
    grad_b = grad_loss_bias(pred, target)
    W = W-grad_w*learning_rate
    b = b-grad_b*learning_rate

def eval(x):
    global W, b

    y = np.matmul(x, W) + b
    return softmax(y)

for i in range(4001):
    if i % 1000 == 0:
        learning_rate = learning_rate/10
    for x_, y_ in zip(x0, t0):
        fit(x_.reshape(1, -1), y_.reshape(1, -1))


from random import randrange

test_size = 50
err = 0
for i in range(test_size):
    test_idx = randrange(0, len(x0)-1)
    test = eval(x0[test_idx])
    print('test #{}: eval={} loss={}'.format(i, test, loss(test, t0[test_idx])))
    sel_test = np.argmax(test)
    sel_tar = np.argmax(t0[test_idx])
    print('test: {}  target: {}\n'.format(sel_test, sel_tar))
    if sel_test != sel_tar:
        err = err + 1

print(1-err/test_size)
print('saving...')
np.savez('./model', w=W, b=b)
print('done')