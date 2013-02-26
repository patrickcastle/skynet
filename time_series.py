import time
import numpy
import csv
import scipy.optimize
import matplotlib
import matplotlib.pyplot
import matplotlib.path
import bayesian_regression
import urllib
import json


drawbridge_access_token = 'a02a3d0776583a0371f73b911a14cab5d84d09e0bd4d6c3fa940be8e363e0b0a4846cead1d5c266ec04d40756fc7deae84e20b8ac21dbcaabf6d700459e6f83c'
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


def offsetted(parsed, granularity=3600, relevant_columns=[1]):
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
            offset = granularity * i
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


def graph_decay(decay, observations):
    line = numpy.zeros(observations.shape[0])
    for i, o in enumerate(observations):
        line[i] = numpy.exp(decay[0] + i * decay[1])
    matplotlib.pyplot.plot(line)
    matplotlib.pyplot.plot(observations)
    matplotlib.pyplot.show()


def find_decay(observations):
    nonzero_indices = numpy.where(observations!=0)[0]
    nonzero = observations[nonzero_indices]
    logged = numpy.log(nonzero)

    design = numpy.ones((nonzero_indices.shape[0],2))
    for row in design:
        row[0] = 10
        row[1] = -0.1
    for i in range(nonzero_indices.shape[0]):
        design[i,1] = nonzero_indices[i]

    mean = numpy.array((0,0))
    covariance = numpy.ones((2,2)) * 100000000 + numpy.identity(2) # so it's not singular
    fit_apply = None
    for i, observation in enumerate(logged):
        fit_apply, mean, covariance = bayesian_regression.fit(numpy.array([logged[i]]), design[i].reshape(1,2), mean, covariance)
    return mean, covariance, fit_apply


def collect_streaming(observations):
    history = []
    for i in range(observations.shape[0]):
        sofar = observations[0:i]
        mean, covariance, fit_apply = find_decay(sofar)
        history.append((mean,covariance))
    return mean, covariance, fit_apply, history


def test_parse():
    parsed = fileparse('series.csv')
    offsetted_vals = offsetted(parsed, relevant_columns=[1])
    return offsetted_vals


def should_restart(observations):
    if len(observations) < 48: # hour step, 2 days
        return False
    else:
        decay, covariance, fit_apply = find_decay(observations)
        # return if not sure yet
        if 0 != len(numpy.where(covariance > 0.1)[0]):
            return False

        half_point = - (decay[0] / decay[1])
        if len(observations) > half_point:
            return True
        else:
            return False

def check_campaign(campaign_id, url='https://drawbridge.castleridgemedia.com/'):
    query = json.dumps([{'campaigns':[campaign_id]},{"type":0,"parameters":{"stats":["impressions"],"groups":["campaign", "hour"]}}])
    url += "calculator/stats_calculator.json?query=%s&drawbridge_access_token=%s" % (query, drawbridge_access_token)
    [datafile, headers] = urllib.urlretrieve(url)
    lines = []
    for line in open(datafile, 'r'):
        lines.append(line)

    doc = lines[0]
    print doc
    struct = json.loads(doc)

    dataframe = []
    for row in struct:
        print row
        try:
            start_time = time.strptime(row['hour'], date_format)
            impressions = row['impressions']
            dataframe.append((start_time, impressions))
        except:
    
    dataframe = sorted(dataframe, key=lambda t: t[0])
    offset = offsetted({campaign_id: dataframe})
    
    print offset
    return should_restart(offset.values()[0][:,1])
