import pandas_datareader.data as web
import datetime

start = datetime.datetime(2016, 1, 1)

end = datetime.datetime(2017, 1, 27)

f = web.DataReader('F', 'iex', start, end)

print f
