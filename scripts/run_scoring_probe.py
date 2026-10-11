"""A deliberate initial collision must be observed and rejected by the scorer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_perception_trials import trial_world

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--webots',required=True)
    args=parser.parse_args()
    output=ROOT/'reports/scoring_probe_verified'
    if output.exists():
        parser.error('probe output already exists; preserve its evidence')
    output.mkdir(parents=True)
    text,config=trial_world('standard','red_box',910110101,'catalog',10,output,ROOT/'reports/perception_audit/hsv_settings.json')
    x,y=config['initial_target']
    text=text.replace(f'translation {x} {y} 0.1', 'translation 0.02 0 0.1')
    path=ROOT/'worlds/benchmark_scoring_probe.wbt'
    path.write_text(text,encoding='utf-8')
    result=subprocess.run([args.webots,'--batch','--mode=fast','--no-rendering','--minimize','--stdout','--stderr',str(path)],
                          env=dict(os.environ,WEBOTS_PYTHON_COMMAND=sys.executable,PYTHONIOENCODING='utf-8'),
                          capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    (output/'process.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stdout[-1500:]+result.stderr[-1500:])
    row=json.loads((output/f"{config['trial_id']}.json").read_text())
    if row['success'] or not row['collision'] or row['object_contact_samples']<1:
        raise AssertionError('deliberate collision was not observed and rejected')
    verdict={'verified':True,'probe':'deliberate initial target overlap','success':row['success'],
             'collision':row['collision'],'object_contact_samples':row['object_contact_samples']}
    (output/'verification.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict),flush=True)

if __name__=='__main__':
    main()
