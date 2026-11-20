#!/usr/bin/env python3
"""Synthetic-only GPU numerical smoke check, never used to tune the measure."""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from apsi_comparator import load_config, infer_pairs, score_probabilities
from populism_axis.run_manifest import write_run_manifest

TEXTS = [
 'The corrupt elite have betrayed ordinary people. We alone represent the true will of the people.',
 'Different groups have legitimate interests. Democratic institutions must protect minorities and permit compromise.',
 'The government should reduce taxes and increase investment in public infrastructure.',
 'The garden has flowers and the weather is pleasant today.',
 'We reject the claim that all politicians are corrupt or that only one party speaks for everyone.',
 'The citizens deserve a voice in government, while independent courts must uphold constitutional rights.',
 ('Our citizens face difficult economic circumstances. Parliament must debate competing proposals. ' * 90),
 ('An unaccountable political class has taken power from the people. The people must reclaim their country. ' * 90),
]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 torch.set_num_threads(2); config=load_config(); result={}
 hypotheses=config['topic_hypotheses']+[h['text'] for h in config['hypotheses']]
 for size,spec in config['models'].items():
  tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision']); predictions={}; scores={}; timing={}
  for name,dtype in [('float32',torch.float32),('float16',torch.float16)]:
   model=AutoModelForSequenceClassification.from_pretrained(spec['id'],revision=spec['revision'],dtype=dtype).to('cuda').eval()
   torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
   probabilities=infer_pairs(model,tok,TEXTS,hypotheses,16,'cuda'); torch.cuda.synchronize()
   timing[name]={'seconds':time.monotonic()-start,'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20}
   predictions[name]=probabilities
   scores[name]=[score_probabilities(row[:25],row[25:],config) for row in probabilities]
   del model; torch.cuda.empty_cache()
  import numpy as np
  difference=np.abs(predictions['float32']-predictions['float16'])
  same_abstentions=all(np.isfinite(x['score']) == np.isfinite(y['score']) for x,y in zip(scores['float32'],scores['float16']))
  score_delta=[abs(x['score']-y['score']) for x,y in zip(scores['float32'],scores['float16']) if np.isfinite(x['score']) and np.isfinite(y['score'])]
  result[size]={'max_probability_difference':float(difference.max()),'mean_probability_difference':float(difference.mean()),'same_abstentions':same_abstentions,'max_score_difference':max(score_delta,default=0),'timing':timing}
  result[size]['float16_pass']=bool(difference.max() <= .01 and same_abstentions and max(score_delta,default=0)<=.02)
  print(size,result[size],flush=True)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 (a.output_dir/'precision_check.json').write_text(json.dumps(result,indent=2)+'\n')
 write_run_manifest(a.output_dir,script='scripts/check_apsi_precision.py',inputs={'script':Path(__file__),'config':Path(__file__).resolve().parents[1]/'anchors/apsi_comparator.json'},extra={'synthetic_texts':TEXTS,'results':result})

if __name__=='__main__': main()
