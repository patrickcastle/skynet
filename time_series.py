import time
import numpy
import csv
import scipy.optimize
import matplotlib
import matplotlib.pyplot
import bayesian_regression

date_format = '%Y-%m-%d %H:%M:%S'
default_keys = ['start_time','hour','impressions','campaign_id','campaign']


def fileparse(filename):
    reader = csv.reader(open(filename, 'r'))
    keys = reader.next()
    return parse(reader, keys)


def parse(iterable, keys=default_keys):
    lists = {}

    try:
        start_time_index = keys.index('start_time')
    except:
        start_time_index = None
    try:
        hour_index = keys.index('hour')
    except:
        hour_index = None
    try:
        impressions_index = keys.index('impressions')
    except:
        impressions_index = None
    try:
        campaign_id_index = keys.index('campaign_id')
    except:
        campaign_id_index = None
    try:
        campaign_index = keys.index('campaign')
    except:
        campaign_index = None

    indices = [hour_index, impressions_index, campaign_id_index, campaign_index]
    indices = filter(lambda i: i != None, indices)

    for line in iterable:
        try:
            line_list = []
            campaign = None
            if None != start_time_index:
                start_time = time.strptime(line[start_time_index], date_format)
                line_list.append(start_time)
            if None != hour_index:
                hour = time.strptime(line[hour_index], date_format)
                line_list.append(hour)
            if None != impressions_index:
                impressions = line[impressions_index]
                line_list.append(impressions)
            if None != campaign_id_index:
                campaign_id = line[campaign_id_index]
                line_list.append(campaign_id)
            if None != campaign_index:
                campaign = line[campaign_index]
                line_list.append(campaign)
            if None != campaign:
                if not campaign in lists:
                    lists[campaign] = []
                lists[campaign].append(line_list)
        except:
            next
    return lists


def offsetted(parsed, granularity=600, relevant_columns=[1]):
    offset_hash = {}
    for category in parsed.keys():
        rows = parsed[category]
        baseline = min([time.mktime(row[0]) for row in rows])
        offsetted_rows = []
        for row in rows:
            new_row = list(row)
            new_row[0] = time.mktime(new_row[0]) - baseline
            offsetted_rows.append(new_row)


        span = int(max([row[0] for row in offsetted_rows])) / granularity + 1 # needs another slot for zero, same condition applies for any length
        arr = numpy.zeros((span, len(relevant_columns) + 1)) # + 1 for offset value
    
        offsetted_rows = sorted(offsetted_rows, key=lambda r: r[0])
        for i in range(span):
            offset = 600 * i
            arr[i, 0] = offset
            index = sublist_index(offsetted_rows, 0, offset)
            for rel_index, rel in enumerate(relevant_columns):
                if -1 == index:
                    arr[i, rel_index + 1] = 0
                else:
                    val = offsetted_rows[index][rel]
                    arr[i, rel_index + 1] = val
        offset_hash[category] = arr
    return offset_hash


def sublist_index(list_of_lists, index, value):
    lo = 0
    hi = len(list_of_lists)
    while lo < hi:
        mid = (lo + hi) / 2
        midval = list_of_lists[mid][index]
        if midval < value:
            lo = mid + 1
        elif midval > value:
            hi = mid
        else:
            return mid
    return -1


# this doesn't work yet
def global_list(parsed, granularity=600, relevant_column=1):
    offset_hash = {}

    # find the dimensions
    baselines = []
    toplines = []
    for category in parsed.keys():
        rows = parsed[category]
        baselines.append(min([time.mktime(row[0]) for row in rows]))
        toplines.append(max([time.mktime(row[0]) for row in rows]))
    baseline = min(baseline)             
    topline = max(toplines)
    span = (topline - baseline) / granularity + 1

    campaigns -= len(parsed.keys())
        
    # make the matrix
    mat = numpy.zeros((eleventy, span))
    for i, category in enumerate(parsed.keys()):
        rows = parsed[category]
        for row in enumerate(rows):
            row_time = time.mktime(row[0])
            col = (row_time - baseline) / granularity
    


def load_it_all(filename, keys=default_keys):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        discard = reader.next() # get rid of headers
        return offsetted(parse(reader, keys))


def offset_factors(filename, keys=default_keys):
    res = load_it_all(filename, keys)

    max_width = max([res[key].shape[0] for key in res.keys()])
    num_obs = len(res.keys())

    mat = numpy.zeros((num_obs, max_width))
    for i, key in enumerate(res.keys()):
        rows = res[key]
        for j, row in enumerate(rows):
            mat[i,j] = row[1]

    return mat


def make_lag_vector(filename):
    res = load_it_all(filename)


def find_and_graph_decay(observations):
    decay = find_decay(observations)
    graph_decay(decay, observations)


def find_decay(observations):
    nonzero_indices = numpy.where(observations!=0)[0]
    nonzero = observations[nonzero_indices]

    num_observations = nonzero.shape[0]
    design = numpy.ones((num_observations,2)) # intercept and decay
    for i in range(num_observations):
        design[i,1] = nonzero_indices[i]
    logged = numpy.log(nonzero) # to prevent NaN all over the goddamn place
    return numpy.linalg.lstsq(design, logged)[0]


def graph_decay(decay, observations):
    line = numpy.zeros(observations.shape[0])
    for i, o in enumerate(observations):
        line[i] = numpy.exp(decay[0] + i * decay[1])
    matplotlib.pyplot.plot(line)
    matplotlib.pyplot.plot(observations)
    matplotlib.pyplot.show()


def find_decay_function(impression_matrix, kernel=(0,0,0)):
    def kernel_error(kernel):
        errors = []
        for i, row in enumerate(impression_matrix):
            for column_index, column in enumerate(row):
                expected = kernel[0]
                for j in range(1, len(kernel)):
                    expected += kernel[j] * column_index ** j
                errors.append((expected - column) ** 2)

        return sum(errors)

    return scipy.optimize.minimize(kernel_error, kernel)


def find_reasonable_decay_length(observations):
    nonzero_indices = numpy.where(observations!=0)[0]
    nonzero = observations[nonzero_indices]
    logged = numpy.log(nonzero)

    design = numpy.ones((nonzero_indices.shape[0],2))
    for i in range(nonzero_indices.shape[0]):
        design[i,1] = nonzero_indices[i]

    mean = numpy.array((1,1))
    covariance = numpy.identity(2)
    fit_apply = None
    for i, observation in enumerate(logged):
        fit_apply, mean, covariance = bayesian_regression.fit(numpy.array([logged[i]]), design[i].reshape(1,2), mean, covariance)
    return fit_apply, mean, covariance


def find_reasonable_and_graph(observations):
    decay = find_reasonable_decay_length(observations)[1]
    nonzero = observations[numpy.where(observations!=0)]
    graph_decay(decay, nonzero)


def test_parse():
    parsed = fileparse('series.csv')
    offsetted_vals = offsetted(parsed, relevant_columns=[1])
    return offsetted_vals
    
