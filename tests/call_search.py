import json, urllib.request, sys
body={'query':'wireless earphones','max_price':5000,'platforms':['Amazon'],'min_rating':3.0,'limit':5}
data=json.dumps(body).encode('utf-8')
req=urllib.request.Request('http://127.0.0.1:8000/search', data=data, headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(r.read().decode('utf-8'))
except Exception as e:
    print('REQUEST_FAILED')
    print(str(e))
