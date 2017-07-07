import time
import datetime
import csv
import sys

"""class save_data:
    def import_data(self, data):
        data = []
        data.append(data)
        return data

    def __iter__(self):
        last_time = time.time()
        count = 0
        data = []
        while True:
            if count == 1:
                self.write_data(data)
                count = 0
                data = []
            else:
                count += 1
            num = self.import_data()
            date = time.strftime("%Y%m%d - %H:%M:%S,")
            date += "%s," %(time.time() - last_time)
            last_time = time.time()
            data.append(date,num)

    def write_data(self, data):
        string = ""
        for date,num in data:
            string += "%s%s\n" % (date, (",")).join([str(n) for n in num])
        with open('data.csv', 'wb') as f:
            f.write(string)"""

"""class save_data:

    def import_data(self, data):
        data = []
        data.append(data)
        return data

    def write_data(self, data):
        string = ""
        num = self.import_data()
        for num in data:
            string += "%s%s\n" % (",").join([str(n) for n in num])
        with open('data.csv', 'wb') as f:
            f.write(string)

def Main():
    data = [28,443,787,654]
    save_data.write_data(data)"""


a = [1,2,3,4,5,6,7,8,9]
last_time = time.time()
data = []
num = []
num.append(a)
date = time.strftime("%Y%m%d - %H:%M:%S,")
date += "%s," %(time.time() - last_time)
last_time = time.time()
data.append((date,num))
string = ""
for date,num in data:
    string += "%s\n" % (date, (",")).join([str(n) for n in num])
with open("output.csv", "wb") as f:
    f.write(string)
print string
