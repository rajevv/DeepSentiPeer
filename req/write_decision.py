import os
import pandas as pd 
import json
import math

data = pd.read_csv('./ICLR-2018.csv', usecols=['Paper_ID', 'Decision'])
print(data.columns)
data.fillna(0)

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


path = './2018/reviews/'
for i,_ in enumerate(range(len(data))):
	print(i)
	id = data.iloc[i]['Paper_ID']
	dcsn = data.iloc[i]['Decision']
	with open(os.path.join(path, id + '.json'), 'r') as f:
		d = json.load(f)
	print(dcsn)
	if dcsn != None and type(dcsn) != float:	
		if 'Accept' in dcsn or 'accept' in dcsn or 'Poster' in dcsn or 'poster' in dcsn or 'invite' in dcsn or 'Invite' in dcsn:
			d['accepted'] = True
		else:
			d['accepted'] = False
	else:
		d['accepted'] = False
	print(d.keys())
	print(d['accepted'])
	with open(os.path.join(path, id + '.json'), 'w') as f:
		json.dump(d, f)