#!/usr/bin/python

import sys, cgi, cgitb, datetime

cgitb.enable(display=0, logdir="pylogs")

formdata = cgi.FieldStorage()

workerid = formdata.getvalue("workerid", "WORKERID_NULL")
thisTrial = formdata.getvalue("this_trial", "THIS_TRIAL_NULL")
Image = formdata.getvalue("Image",      "IMAGE_NULL")
ImagePath = formdata.getvalue("ImagePath", "ImagePath_NULL")
category = formdata.getvalue("category", "category_NULL")
imageStart = formdata.getvalue("imageStart", "imageStart_NULL")
imageEnd = formdata.getvalue("imageEnd", "imageEnd_NULL")
response = formdata.getvalue("response", "response_NULL")
responseTime = formdata.getvalue("responseTime", "responseTime_NULL")
currentTime = formdata.getvalue("currentTime", "currentTime_NULL")
ButtonLeft = formdata.getvalue("ButtonLeft", "ButtonLeft_NULL")
ButtonRight = formdata.getvalue("ButtonRight", "ButtonRight_NULL")

output_fname = 'data/catPer_{}.txt'.format(workerid)
timestamp_str = '{0}'.format( datetime.datetime.utcnow() )
filehandle = open(output_fname, 'a')
to_print = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n'.format(
           workerid, thisTrial, Image, ImagePath, 
           category, imageStart, imageEnd, response, 
           responseTime, currentTime,ButtonLeft,ButtonRight)

filehandle.write( to_print )
filehandle.close()

sys.stdout.write('Content-type: text/plain; charset=UTF-8\n\n')
sys.stdout.write('Done.')
