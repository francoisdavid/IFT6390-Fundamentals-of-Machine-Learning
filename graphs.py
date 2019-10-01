import numpy as np
import matplotlib.pyplot as plt

def q5():
    #  values from print command from solution.py get_test_errors (function runs on validation data first to get the below values)
    er_h = [0.66666667, 0.66666667, 0.66666667, 0.23333333, 0.1, 0.3, 0.66666667, 0.66666667, 0.66666667]
    h = [1.0e-03, 1.0e-02, 1.0e-01, 3.0e-01, 1.0e+00, 3.0e+00, 1.0e+01, 1.5e+01, 2.0e+01]
    er_s = [0.66666667, 0.16666667, 0.06666667, 0.03333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333]
    sigma = h

    # er_h = [0.71428571, 0.71428571, 0.71428571, 0.25, 0.10714286, 0.25, 0.64285714, 0.64285714, 0.64285714]
    # h = [1.0e-03, 1.0e-02, 1.0e-01, 3.0e-01, 1.0e+00, 3.0e+00, 1.0e+01, 1.5e+01, 2.0e+01]

    # er_s = [0.71428571, 0.17857143, 0.07142857, 0.03571429, 0.14285714, 0.14285714, 0.42857143, 0.42857143, 0.42857143]
    # sigma =  [1.0e-03, 1.0e-02, 1.0e-01, 3.0e-01, 1.0e+00, 3.0e+00, 1.0e+01, 1.5e+01, 2.0e+01]

    plt.plot(h, er_h, label="Hard Parzen")
    plt.plot(sigma, er_s, label="Soft Parzen")
    plt.xlabel(r'$h, \sigma$')
    plt.ylabel("error rate")
    plt.legend(fancybox=True, framealpha=0.5)

    plt.show()

    plt.plot(h, er_h, label="Hard Parzen")
    plt.plot(sigma, er_s, label="Soft Parzen")
    plt.xlabel("log-scaled values of "+r'$h, \sigma$')
    plt.ylabel("error rate")
    plt.legend(fancybox=True, framealpha=0.5)
    plt.xscale('log')

    plt.show()

def q9(): 
    h = np.array([0.001,0.01,0.1,0.3,1.0,3.0,10.0,15.0,20.0])

    hard_avgs = np.array([0.6665999999999999, 0.662, 0.4134, 0.17193333333333333, 0.1750666666666667, 0.37739999999999996, 0.6650666666666665, 0.6666666666666665, 0.6666666666666665]     )
    hard_stds = np.array([0.0014892205269125779, 0.014772346537440221, 0.11506711085275409, 0.09073181482932116, 0.11854908031884703, 0.18828558686798683, 0.016114038048305027, 1.1102230246251565e-16, 1.1102230246251565e-16])
    soft_avgs = np.array([0.6075333333333333, 0.15933333333333335, 0.1218, 0.12586666666666668, 0.1704666666666667, 0.18406666666666666, 0.1784, 0.18513333333333334, 0.18133333333333335] )
    soft_stds = np.array([0.05841350680945098, 0.09390065672471803, 0.09382184062241465, 0.08683204733532686, 0.10610793247129703, 0.1066953919654765, 0.10039973439539901, 0.10534116004677374, 0.0988286957877676])

    # er_h = [0.66666667, 0.66666667, 0.66666667, 0.23333333, 0.1, 0.3, 0.66666667, 0.66666667, 0.66666667]
    # h = [1.0e-03, 1.0e-02, 1.0e-01, 3.0e-01, 1.0e+00, 3.0e+00, 1.0e+01, 1.5e+01, 2.0e+01]
    # er_s = [0.66666667, 0.16666667, 0.06666667, 0.03333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333, 0.13333333]
    # sigma = h

    # hard_avgs = [0.6665999999999999, 0.662, 0.40886666666666666, 0.17073333333333335, 0.17673333333333333, 0.37760000000000005, 0.6649333333333332, 0.6666666666666665, 0.6666666666666665]    
    # soft_avgs = [0.6083333333333333, 0.159, 0.12693333333333334, 0.12853333333333336, 0.16853333333333337, 0.17966666666666667, 0.18313333333333334, 0.1846, 0.182]

    # plt.plot(h, hard_avgs, label="Hard Parzen") 
    # plt.plot(h, soft_avgs, label="Soft Parzen")

    plt.errorbar(h, hard_avgs, 0.2*hard_stds, label="Hard Parzen")
    plt.errorbar(h, soft_avgs, 0.2*soft_stds, label="Soft Parzen") 
    plt.xlabel(r'$h, \sigma$')
    plt.ylabel("avg error rate")
    plt.legend(fancybox=True, framealpha=0.5, loc=5)
    plt.show()

    # plt.plot(h, hard_avgs, label="Hard Parzen") 
    # plt.plot(h, soft_avgs, label="Soft Parzen") 
    plt.errorbar(h, hard_avgs, 0.2*hard_stds, label="Hard Parzen")
    plt.errorbar(h, soft_avgs, 0.2*soft_stds, label="Soft Parzen")
    plt.xlabel("log-scaled values of "+r'$h, \sigma$')
    plt.ylabel("avg error rate")
    plt.legend(fancybox=True, framealpha=0.5, loc=9)
    plt.xscale('log')
    plt.show()

q5()
q9()