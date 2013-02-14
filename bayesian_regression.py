import numpy
import numpy.linalg
import numpy.random


def fit(target, training, mean=None, covariance=None, noise=20):
    if None == mean:
        mean = numpy.zeros(training.shape[1])
    if None == covariance:
        covariance = numpy.identity(training.shape[1])

    print "target " + str(target)

    covariance_next = numpy.linalg.inv(numpy.linalg.inv(covariance) + noise * numpy.dot(training.T, training))
    print "training shape " + str(training.T.shape)
    print "target shape " + str(target.shape)
    print "covariance shape " + str(covariance.shape)
    print "covariance next " + str(covariance_next.shape)
    mean_next = numpy.dot(covariance_next, (numpy.dot(numpy.linalg.inv(covariance), mean) + noise * numpy.dot(training.T, target)))
    
    numpy.dot(covariance_next, numpy.dot(covariance, mean) + numpy.dot(target.T, target))
    
    def fit_apply(test):
        numpy.dot(mean_next, test.T)

    return [fit_apply, mean_next, covariance_next]
