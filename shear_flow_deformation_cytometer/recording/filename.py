from datetime import datetime, time
import os


def path(path, suffix):
    date_time = datetime.now()
    flname = str(date_time.year) + '_' + \
                 '{0:02d}'.format(date_time.month) + '_' + \
                 '{0:02d}'.format(date_time.day) + '_' + \
                 '{0:02d}'.format(date_time.hour) + '_' + \
                 '{0:02d}'.format(date_time.minute) + '_' + \
                 '{0:02d}'.format(date_time.second) + suffix

    flpath = path + '\\' + flname + '.tif'
    return flpath


def Conpath(bpath):
    p = bpath[:-4] + '_config' + '.txt'
    return p


def Dpath():
    now = datetime.now()
    folderName =  str(now.year) + '.' + str(now.month) + '.' + str(now.day)
    path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') + '\\' + folderName
    return path