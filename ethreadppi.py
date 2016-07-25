import sys
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

## Usage example :
# python ethreadppi.py 12asB-hhsearch.out 12asB-raptorx.rank 12asB-sparksx.zs12 0 ./model_data/model_mcc_homo.pkl ./model_data/model_isc_homo.pkl
if len(sys.argv) != 7:
    print "USAGE : python ethreadppi.py <hhsearch>"
    print "                              <raptorx>"  
    print "                              <sparksx>"
    print "                   <1/0> homo:0 , hete:1"
    print "                         <rfc_model_mcc>"
    print "                      <rfc_model_isc>"
    sys.exit (1)
    

hhsearch_f = sys.argv[1]
raptor_f = sys.argv[2]
sparks_f = sys.argv[3]


input_hh = open(hhsearch_f)
input_rx = open(raptor_f)
input_sp = open(sparks_f)

lines_hh = input_hh.readlines()
lines_rx = input_rx.readlines()
lines_sp = input_sp.readlines()

input_hh.close()
input_rx.close()
input_sp.close()

hhsearch_val = []
hhsearch_key = []
for line in lines_hh:
    words = line.split()
    no_words = len(words)
    if no_words == 11 : 
        try:
            z1 = int(words[0])
            z2 = int(words[7])
            z3 = float(words[2])
            hhsearch_val.append(z3)
            hhsearch_key.append(words[1])
        except ValueError:
            continue
raptor_val = []
raptor_key = []
for line in lines_rx:
    words = line.split()
    no_words = len(words)
    if no_words == 11 : 
        try:
            z1 = int(words[0])
            z2 = int(words[10])
            z3 = float(words[3])
            raptor_val.append(z3)
            raptor_key.append(words[1])
        except ValueError:
            continue

sparks_key = []; sparks_val = []
for line in lines_sp[0:]:
    words = line.split()
    sparks_key.append(words[1])
    sparks_val.append(float(words[2]))

# For homodimer dimer_type = 0 
# For heterodimer dimer_type = 1

dimer_type = int(sys.argv[4])
if dimer_type == 0 :
    hh_mean = 75.651 ; hh_std = 40.352;
    rx_mean = 100.960 ; rx_std = 78.954;
    sp_mean = 2.212 ; sp_std = 1.578;
else :
    hh_mean = 71.966 ; hh_std = 41.088;
    rx_mean = 117.378 ; rx_std = 86.699;
    sp_mean = 2.412 ; sp_std = 1.597;


data_hh = {'template':hhsearch_key,'score_hh':hhsearch_val}
data_rx = {'template':raptor_key,'score_rx':raptor_val}
data_sp = {'template':sparks_key,'score_sp':sparks_val}

frame_hh = pd.DataFrame(data_hh, columns=['template','score_hh'])
frame_rx = pd.DataFrame(data_rx, columns=['template','score_rx'])
frame_sp = pd.DataFrame(data_sp, columns=['template','score_sp'])

frame_hh['zscore_hh'] = (frame_hh.score_hh - hh_mean)/ hh_std
frame_rx['zscore_rx'] = (frame_rx.score_rx - rx_mean)/ rx_std
frame_sp['zscore_sp'] = (frame_sp.score_sp - sp_mean)/ sp_std

hh_rx = pd.merge(left=frame_hh,right=frame_rx,on='template',how='inner').fillna(0)
hh_rx_sp = pd.merge(left=hh_rx,right=frame_sp,on='template',how='inner').fillna(0)

print hh_rx_sp.head()
def load(ifn):
    return pd.read_csv(ifn, header=None, delim_whitespace=True)

def predict_mcc(X):
    with open( sys.argv[5], 'r') as f:
        clf = pickle.load(f)        
    predictions = clf.predict_proba(X)
    return predictions[:,1]

def predict_isc(X):
    with open( sys.argv[6], 'r') as f:
        clf = pickle.load(f)        
    predictions = clf.predict_proba(X)
    return predictions[:, 1]

X1=hh_rx_sp[[2,4,6]].values
temp_id=hh_rx_sp['template'].values
predicted_mcc = predict_mcc(X1)
predicted_isc = predict_isc(X1)

data_mcc = {'template':temp_id,'mcc':predicted_mcc, 'isc':predicted_isc }
frame_mcc = pd.DataFrame (data_mcc, columns=['template','mcc','isc'])
print frame_mcc.to_string(index=False)
