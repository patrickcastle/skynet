import numpy
import numpy.linalg
import numpy.random


def fit(target, training, mean=None, covariance=None, noise=0.05):
    if None == mean:
        mean = numpy.zeros(training.shape[1])
    if None == covariance:
        covariance = numpy.identity(training.shape[1])

    covariance_next = numpy.linalg.inv(numpy.linalg.inv(covariance) + noise * numpy.dot(training.T, training))
    mean_next = numpy.dot(covariance_next, (numpy.dot(numpy.linalg.inv(covariance), mean) + noise * numpy.dot(training.T, target)))
    
    numpy.dot(covariance_next, numpy.dot(covariance, mean) + numpy.dot(target.T, target))
    
    def fit_apply(test):
        numpy.dot(mean_next, test.T)

    return [fit_apply, mean_next, covariance_next]
