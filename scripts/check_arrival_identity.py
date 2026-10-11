"""Post-audit conservative identity review, preserving the original frozen scores."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from visionlab.evaluation import circle_box_gap, score_trial

def conservative_score(samples, objects, case, duration):
    reviewed=[]
    for packet in samples:
        other_gap=min(circle_box_gap(packet['robot_position'],objects[name]) for name in ('DISTRACTOR_A','DISTRACTOR_B'))
        packet=dict(packet)
        # Arrival must be nearer to the intended object than to either distractor.
        packet['target_visible']=packet['target_visible'] and packet['target_gap_m']+0.001<other_gap
        reviewed.append(packet)
    result=score_trial(reviewed,case,0.032,duration)
    original=score_trial(samples,case,0.032,duration)
    result['success']=result['success'] and not original['collision']
    return result

def compute(directory):
    summary=json.loads((directory/'summary.json').read_text())
    rows=[]
    for row in summary['trials']:
        world=ROOT/'worlds'/f"benchmark_{row['trial_id']}.wbt"
        text=world.read_text(encoding='utf-8')
        if hashlib.sha256(text.encode()).hexdigest()!=row['world_sha256']:
            raise ValueError('world does not match recorded trial')
        objects={}
        for name in ('DISTRACTOR_A','DISTRACTOR_B'):
            match=re.search(r'DEF '+name+r' Solid \{\s+translation ([^\n]+)',text)
            if not match:
                raise ValueError('missing scored object in saved world')
            objects[name]=list(map(float,match.group(1).split()))
        with gzip.open(directory/f"{row['trial_id']}.telemetry.jsonl.gz",'rt',encoding='utf-8') as stream:
            samples=[json.loads(line) for line in stream]
        result=conservative_score(samples,objects,row['case'],row['duration'])
        rows.append({'trial_id':row['trial_id'],'case':row['case'],'target':row['target'],'backend':row['backend'],
                     'original_success':row['success'],'conservative_success':result['success'],
                     'conservative_arrival_sim_seconds':result['arrival_sim_seconds'], 'object_geometry':objects,
                     'world_sha256':row['world_sha256']})
    return {'protocol':'Post-audit conservative review: require selected target to be at least 1 mm nearer than either distractor throughout the 0.5-second stop; original frozen scores remain preserved.',
            'scope':'Approximate planar circle/box distances from recorded robot positions and hash-verified world geometry; no controller or detector changes.',
            'algorithm_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'trials':rows}

def main():
    directory=ROOT/'reports/closed_loop_verified'
    result=compute(directory)
    (directory/'identity_review.json').write_text(json.dumps(result,indent=2)+'\n')
    changes=[r for r in result['trials'] if r['original_success']!=r['conservative_success']]
    print(json.dumps({'reviewed_trials':len(result['trials']),'changed':changes},indent=2))

if __name__=='__main__':
    main()
