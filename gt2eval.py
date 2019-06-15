import json
import string

j = json.load(open('/data2/activity_net/captions/val_2.json'))
j1 = {"version": "VERSION 1.0", "results": {}}

for k,v in j.items():
    j1["results"][k] = []
    for i,t in enumerate(v['timestamps']):
        sent = v["sentences"][i]
        sent = str(sent.encode('ascii','ignore').encode('utf-8')).lower()
        sent = sent.replace(',', ' ').translate(None, string.punctuation).strip()
        info = {"timestamp" : t, "sentence" : sent}
        j1["results"][k].append(info)
