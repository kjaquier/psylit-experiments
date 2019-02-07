#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'psylit-experiments'))
	print(os.getcwd())
except:
	pass

#%%
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata
import flair


#%%
text = strip_headers(load_etext(2701)).strip()
with open("mobydick.txt","w+") as f:
    f.write(text)
print("done")


#%%
import json
data = json.load(open("gutenberg-response.json"))


#%%
import pandas as pd
def extract(d):
    r = {}
    r['languages'] = d['languages']
    r['id'] = d['id']
    r['title'] = d['title']
    births = [a.get('birth_year') for a in d['authors']]
    death = [a.get('death_year') for a in d['authors']]
    min_birth = min(filter(bool, births), default=99999)
    max_death = max(filter(bool, death), default=-1)
    
    r['year_range'] = [min_birth, max_death]
    return r
            
res  =[extract(x) for x in data['results']]
df = pd.DataFrame(data['results'])


#%%
from collections import Counter
print("from\n", Counter(x['year_range'][0] for x in res))
print("to", Counter(x['year_range'][1] for x in res))


#%%



#%%



