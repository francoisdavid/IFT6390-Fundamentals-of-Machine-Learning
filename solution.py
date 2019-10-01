import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        features = iris[:,:4]
        feature_means = np.mean(features, axis=0)
        return feature_means

    def covariance_matrix(self, iris):
        features = iris[:,:4]
        cov = np.cov(features.T)
        return cov

    def feature_means_class_1(self, iris):
        iris_1 = iris[np.where(iris[:,-1]==1)]
        return self.feature_means(iris_1)

    def covariance_matrix_class_1(self, iris):
        iris_1 = iris[np.where(iris[:,-1]==1)]
        return self.covariance_matrix(iris_1)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.X = train_inputs
        self.Y = train_labels.astype(int)
        # return self.X, self.Y

    def one_hot(self, a):
        b = np.zeros(shape=(np.max(self.Y)))
        b[a-1] = 1
        return np.array(b)

    def compute_predictions(self, test_data):
        test_y = []
        # print("T: ", test_data)
        for t in test_data:
            # print("t: ", t)
            # print("X: ", self.X)
            # print("Y: ", self.Y)
            # print("h: ", self.h)
            # print("self.X-t: ", self.X-t)
            test_y_1hot = []    
            dists = np.sqrt(np.sum((self.X-t)**2, axis=1))
            # print("dists: ", dists)
            I = dists < self.h
            if (len(I)==0):
                test_y.append(draw_rand_label(t, self.label_list))
                continue
            # print("I: ", I)
            y = I*self.Y
            y = y[np.where(y>0)]
            # print("y: ", y)
            for i in range(len(y)):
                test_y_1hot.append(self.one_hot(y[i]))
            # print("test_y_1hot: ", test_y_1hot)
            # print("test_y_1hot sum: ", np.sum(test_y_1hot, axis=0))
            test_y.append(np.argmax(np.sum(test_y_1hot, axis=0))+1)
            # print("predicted test_y: ", test_y)
        print("final test_y: ", test_y)
        return test_y


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def one_hot(self, a):
        b = np.zeros(shape=(np.max(self.Y)))
        b[a-1] = 1
        return np.array(b)

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.X = train_inputs
        self.Y = train_labels.astype(int)

    def compute_predictions(self, test_data):
        test_y = []
        # print("T: ", test_data)
        for t in test_data:
            # print("t: ", t)
            # print("X: ", self.X)
            # print("Y: ", self.Y)
            # print("sigma: ", self.sigma)
            # print("self.X-t: ", self.X-t)
            test_y_1hot = [self.one_hot(i) for i in self.Y]    
            dists = np.sqrt(np.sum((self.X-t)**2, axis=1))
            # print("dists: ", dists)
            # I = dists < self.h
            # print("I: ", I)
            d = self.X.shape[1]
            # print("d: ", d)
            K = (1/(((2*np.pi)**(d/2))*self.sigma**d)) * np.exp((-1/2)*(np.square(dists)/self.sigma**2))
            K = np.expand_dims(K, axis=1)
            # print("K: ", K)
            y = K*test_y_1hot
            # y = y[np.where(y>0)]
            # print("y: ", y)
            # for i in range(len(y)):
            #     test_y_1hot.append(self.one_hot(y[i]))
            # print("test_y_1hot: ", test_y_1hot)
            # print("test_y_1hot sum: ", np.sum(test_y_1hot, axis=0))
            test_y.append(np.argmax(np.sum(y, axis=0))+1)
            # print("predicted test_y: ", test_y)
        print("final test_y: ", test_y)
        return test_y



def split_dataset(iris):
    indices = np.arange(len(iris))
    train_ix = indices[np.where(indices%5 < 3)]
    valid_ix = indices[np.where(indices%5 == 3)]
    test_ix = indices[np.where(indices%5 == 4)]
    return (iris[train_ix], iris[valid_ix], iris[test_ix])


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        h = HardParzen(h)
        h.train(self.x_train, self.y_train)
        pred_y = h.compute_predictions(self.x_val)
        # print(pred_y)
        # print(self.y_val)
        incorrect_preds = np.sum(pred_y != self.y_val)
        print("hard error rate: ", incorrect_preds/len(self.y_val))
        return incorrect_preds/len(self.y_val)

    def soft_parzen(self, sigma):
        s = SoftRBFParzen(sigma)
        s.train(self.x_train, self.y_train)
        pred_y = s.compute_predictions(self.x_val)
        # print(pred_y)
        # print(self.y_val)
        incorrect_preds = np.sum(pred_y != self.y_val)
        print("soft error rate: ", incorrect_preds/len(self.y_val))
        return incorrect_preds/len(self.y_val)
    
    def predict_test_hard_parzen(self, test_x, test_y, h):
        h = HardParzen(h)
        h.train(self.x_train, self.y_train)
        pred_y = h.compute_predictions(test_x)
        incorrect_preds = np.sum(pred_y != test_y)
        print("hard test error rate: ", incorrect_preds/len(test_y))
        return incorrect_preds/len(test_y)

    def predict_test_soft_parzen(self, test_x, test_y, sigma):
        h = SoftRBFParzen(sigma)
        h.train(self.x_train, self.y_train)
        pred_y = h.compute_predictions(test_x)
        incorrect_preds = np.sum(pred_y != test_y)
        print("soft test error rate: ", incorrect_preds/len(test_y))
        return incorrect_preds/len(test_y)


def get_test_errors(iris):
    # runs on valid dataset first for plotting the graph required for practical report, and then runs on test data
    train, valid, test = split_dataset(iris)
    print("test shape: ", test.shape)
    er = ErrorRate(train[:,:4],train[:,4],valid[:,:4],valid[:,4])
    h = np.array([0.001,0.01,0.1,0.3,1.0,3.0,10.0,15.0,20.0])
    er_h = []
    sigma = np.array([0.001,0.01,0.1,0.3,1.0,3.0,10.0,15.0,20.0])
    er_s = []
    for i in range(len(h)):
        er_h.append(er.hard_parzen(h[i]))
        er_s.append(er.soft_parzen(sigma[i]))
    er_h = np.array(er_h)
    er_s = np.array(er_s)
    print("er_h: ", er_h)
    print("h: ", h)
    print("er_s: ", er_s)
    print("sigma: ", sigma)
    
    '''
    FOR REPORT

    h:  [1.0e-03 1.0e-02 1.0e-01 3.0e-01 1.0e+00 3.0e+00 1.0e+01 1.5e+01 2.0e+01]
    er_s:  [0.66666667 0.16666667 0.06666667 0.03333333 0.13333333 0.13333333
    0.13333333 0.13333333 0.13333333]
    sigma:  [1.0e-03 1.0e-02 1.0e-01 3.0e-01 1.0e+00 3.0e+00 1.0e+01 1.5e+01 2.0e+01]
    h_star:  1.0
    sigma_star:  0.3

    er_h:  [0.71428571 0.71428571 0.71428571 0.25       0.10714286 0.25
    0.64285714 0.64285714 0.64285714]
    h:  [1.0e-03 1.0e-02 1.0e-01 3.0e-01 1.0e+00 3.0e+00 1.0e+01 1.5e+01 2.0e+01]
    er_s:  [0.71428571 0.17857143 0.07142857 0.03571429 0.14285714 0.14285714
    0.42857143 0.42857143 0.42857143]
    sigma:  [1.0e-03 1.0e-02 1.0e-01 3.0e-01 1.0e+00 3.0e+00 1.0e+01 1.5e+01 2.0e+01]
    h_star:  1.0
    sigma_star:  0.3
    '''

    sort_h_ix = np.argsort(er_h)
    sort_s_ix = np.argsort(er_s)
    er_h = er_h[sort_h_ix]
    h = h[sort_h_ix]
    er_s = er_s[sort_s_ix]
    sigma = sigma[sort_s_ix]
    h_star = h[0]
    sigma_star = sigma[0]
    print("h_star: ", h_star)
    print("sigma_star: ", sigma_star)
    best_soft = er.predict_test_soft_parzen(test[:,:4],test[:,4],sigma_star)
    best_hard = er.predict_test_hard_parzen(test[:,:4],test[:,4],h_star)
    return [best_hard,best_soft]


def random_projections(X, A):
    p = []
    for x in X:
        # print("x: ", x)
        # print("A.T: ", A.T)
        p.append((1/np.sqrt(2)) * np.dot(A.T,x))
    p = np.array(p)
    # print("p shape: ", p.shape)
    return p

def q9(iris):
    train, valid, test = split_dataset(iris)
    train_x = train[:,:4]
    train_y = train[:,4]
    valid_x = valid[:,:4]
    valid_y = valid[:,4]
    h = np.array([0.001,0.01,0.1,0.3,1.0,3.0,10.0,15.0,20.0])
    sigma = h
    hard_avgs = []
    soft_avgs = []
    hard_stds = []
    soft_stds = []
    for param in h:
        hard_errors = []
        soft_errors = []
        for i in range(500):
            A = np.random.normal(loc=0.0, scale=1.0, size=(4,2))
            x_train = random_projections(train_x, A)
            x_val = random_projections(valid_x, A)
            e = ErrorRate(x_train, train_y, x_val, valid_y)
            err_h = e.hard_parzen(param)
            hard_errors.append(err_h)
            err_s = e.soft_parzen(param)
            soft_errors.append(err_s)
        hard_avgs.append(np.mean(hard_errors))
        hard_stds.append(np.std(hard_errors))
        soft_avgs.append(np.mean(soft_errors))
        soft_stds.append(np.std(soft_errors))
    print("hard_avgs: ", hard_avgs)
    print("hard_stds: ", hard_stds)
    print("soft_avgs: ", soft_avgs)
    print("soft_stds: ", soft_stds)

# iris = np.loadtxt("iris.txt")
# print(iris)
# print(iris.shape)

# get_test_errors(iris)
# q9(iris)

# q = Q1()
# print(q.feature_means(iris))
# print(q.covariance_matrix(iris))
# print(q.feature_means_class_1(iris))
# print(q.covariance_matrix_class_1(iris))

# H = HardParzen(0.4)
# t_x = [[0.54340494, 0.27836939, 0.42451759],
#  [0.84477613, 0.00471886, 0.12156912],
#  [0.67074908, 0.82585276, 0.13670659],
#  [0.57509333, 0.89132195, 0.20920212],
#  [0.18532822, 0.10837689, 0.21969749],
#  [0.97862378, 0.81168315, 0.17194101],
#  [0.81622475, 0.27407375, 0.43170418],
#  [0.94002982, 0.81764938, 0.33611195],
#  [0.17541045, 0.37283205, 0.00568851],
#  [0.25242635, 0.79566251, 0.01525497],
#  [0.59884338, 0.60380454, 0.10514769],
#  [0.38194344, 0.03647606, 0.89041156],
#  [0.98092086, 0.05994199, 0.89054594],
#  [0.5769015 , 0.74247969, 0.63018394],
#  [0.58184219, 0.02043913, 0.21002658],
#  [0.54468488, 0.76911517, 0.25069523],
#  [0.28589569, 0.85239509, 0.97500649],
#  [0.88485329, 0.35950784, 0.59885895],
#  [0.35479561, 0.34019022, 0.17808099],
#  [0.23769421, 0.04486228, 0.50543143],
#  [0.37625245, 0.5928054 , 0.62994188],
#  [0.14260031, 0.9338413 , 0.94637988],
#  [0.60229666, 0.38776628, 0.363188  ],
#  [0.20434528, 0.27676506, 0.24653588],
#  [0.173608  , 0.96660969, 0.9570126 ],
#  [0.59797368, 0.73130075, 0.34038522],
#  [0.0920556 , 0.46349802, 0.50869889],
#  [0.08846017, 0.52803522, 0.99215804],
#  [0.39503593, 0.33559644, 0.80545054],
#  [0.75434899, 0.31306644, 0.63403668],
#  [0.54040458, 0.29679375, 0.1107879 ],
#  [0.3126403 , 0.45697913, 0.65894007],
#  [0.25425752, 0.64110126, 0.20012361],
#  [0.65762481, 0.77828922, 0.7795984 ],
#  [0.61032815, 0.30900035, 0.69773491],
#  [0.8596183 , 0.62532376, 0.98240783],
#  [0.97650013, 0.16669413, 0.02317814],
#  [0.16074455, 0.92349683, 0.95354985],
#  [0.21097842, 0.36052525, 0.54937526],
#  [0.27183085, 0.46060162, 0.69616156],
#  [0.5003559 , 0.71607099, 0.52595594],
#  [0.00139902, 0.39470029, 0.49216697],
#  [0.40288033, 0.3542983 , 0.50061432],
#  [0.44517663, 0.09043279, 0.27356292],
#  [0.9434771 , 0.02654464, 0.03999869],
#  [0.28314036, 0.58234417, 0.9908928 ],
#  [0.99264224, 0.99311737, 0.11004833],
#  [0.66448145, 0.52398683, 0.17314991],
#  [0.94296024, 0.24186009, 0.99893227],
#  [0.58269382, 0.183279  , 0.38684542],
#  [0.18967353, 0.41077067, 0.59468007],
#  [0.71658609, 0.48689148, 0.30958982],
#  [0.57744137, 0.44170782, 0.3596781 ],
#  [0.32133193, 0.20820724, 0.45125862],
#  [0.49184291, 0.89907631, 0.72936046],
#  [0.77008977, 0.37543925, 0.34373954],
#  [0.65503521, 0.71103799, 0.11353758],
#  [0.13302869, 0.45603906, 0.15973623],
#  [0.9616419 , 0.83761574, 0.52016069],
#  [0.21827226, 0.13491872, 0.97907035],
#  [0.7070435 , 0.85997556, 0.38717263],
#  [0.25083402, 0.29943802, 0.85689553],
#  [0.47298399, 0.66327705, 0.80572861],
#  [0.2529805 , 0.07957344, 0.73276061],
#  [0.96139748, 0.95380473, 0.49049905],
#  [0.63219206, 0.73299502, 0.9024095 ],
#  [0.16224692, 0.40588132, 0.41709074],
#  [0.69559103, 0.42484724, 0.85811423],
#  [0.84693248, 0.07019911, 0.30175241],
#  [0.97962368, 0.035627  , 0.49239265],
#  [0.95237685, 0.81057376, 0.29433044],
#  [0.59623352, 0.43117785, 0.5923975 ],
#  [0.8937521 , 0.55402119, 0.49286651],
#  [0.31927046, 0.26336578, 0.54228061],
#  [0.08226452, 0.63563671, 0.79640523],
#  [0.95474751, 0.68462427, 0.48829317],
#  [0.48541431, 0.96669292, 0.21134789],
#  [0.41164814, 0.98966558, 0.02841186],
#  [0.70132651, 0.02517156, 0.32088173],
#  [0.07352706, 0.06088456, 0.11140632],
#  [0.16926891, 0.62768628, 0.43839309],
#  [0.83090376, 0.23979219, 0.19005271],
#  [0.71189966, 0.85829493, 0.55905589],
#  [0.70442041, 0.60511204, 0.55921728],
#  [0.86039419, 0.91975536, 0.84960733],
#  [0.25446654, 0.87755554, 0.43513019],
#  [0.72949434, 0.41264077, 0.19083605],
#  [0.70601952, 0.24063282, 0.85132443],
#  [0.82410229, 0.52521179, 0.38634079],
#  [0.59088079, 0.13752361, 0.80827041],
#  [0.96582582, 0.7797958 , 0.23933508],
#  [0.86726041, 0.80811501, 0.06368112],
#  [0.2312283 , 0.58968545, 0.13748695],
#  [0.6784407 , 0.99219069, 0.28575198],
#  [0.76091276, 0.04652717, 0.33253591],
#  [0.94455279, 0.63651704, 0.60184861],
#  [0.92818468, 0.18167941, 0.01782318],
#  [0.19007218, 0.5218718 , 0.49582199],
#  [0.80049121, 0.85943631, 0.21295603],
#  [0.43726884, 0.42161751, 0.05471738]]
# t_y = [
#     3, 3, 4, 4, 2, 4, 3, 4, 4, 1, 4, 1, 2, 2, 4, 3, 3, 1, 4, 2, 3, 2, 1, 2, 4, 4, 2, 3, 2, 3, 3, 3, 4, 2, 3, 3, 2,
#     4, 4, 1, 4, 2, 1, 3, 2, 2, 3, 3, 1, 4, 4, 4, 3, 1, 4, 4, 3, 1, 2, 4, 1, 3, 2, 3, 2, 1, 1, 1, 2, 2, 4, 2, 4, 4,
#     1, 1, 1, 3, 1, 1, 1, 3, 3, 2, 2, 2, 4, 4, 2, 1, 4, 1, 3, 2, 1, 4, 1, 4, 2, 3
# ]
# H.train(t_x, t_y)
# H.compute_predictions(
#     [[0.63508815, 0.22996917 ,0.05120709],
#     [0.02846381 ,0.12284775 ,0.22021252],
#     [0.82902275 ,0.28549183 ,0.78106408],
#     [0.50466581 ,0.13844892 ,0.77803655]]
# )
