import numpy
import numpy.linalg
import numpy.random


def fit(target, training, mean=None, covariance=None):
    if None == mean:
        mean = numpy.zeros(training.shape[1])
    if None == covariance:
        covariance = numpy.ident(traiing.shape[1])

    covariance_next = covariance + numpy.dot(training.T, training)
    mean_next = numpy.dot(numpy.linalg.convariance_next, (numpy.dot(covariance, mean) + numpy.dot(example.T, target)))
    
    def fit_apply(test):
        numpy.dot(mean_next, test.T)

    return [fit_apply, mean_next, covariance_next]
