# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
#__author__= 'ihciah@gmail.com'

import urllib, urllib2, time

def login(username, password):
    url="http://10.108.255.249/include/auth_action.php"
    data={"username": username,
          "password": password,
          "action": "login",
          "ac_id": 1,
          "ajax": 1}
    data=urllib.urlencode(data)
    urllib2.urlopen(url, data)

username="15307130240"
password="fduhly123"

print 'what the fuck???'
while True:
    try:
        print '???'
        login(username, password)
    except:
        print 'Fail...'
        pass
    time.sleep(30)
