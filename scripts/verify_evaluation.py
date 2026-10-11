"""Verify saved audit identities/completeness and recompute telemetry scores."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_perception import generate, AUDIT_SEEDS
from scripts.make_target_data import NAMES
from visionlab.evaluation import score_trial

def check(condition, message):
    if not condition:
        raise AssertionError(message)

def verify_images(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    report = json.loads((directory/'results.json').read_text())
    check(report['protocol'] == protocol, 'image report/protocol mismatch')
    check(protocol['model_sha256'] == hashlib.sha256((ROOT/'models/targets/target_catalog.pt').read_bytes()).hexdigest(), 'model changed')
    for relative, digest in protocol['source_sha256'].items():
        check(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, 'image source changed: '+relative)
    training = json.loads((ROOT/'reports/target_training.json').read_text())
    historical_manifest={'classes':NAMES,'source':'procedural rgb boxes with exact drawn masks','seed':training['seed'],
                         'samples':[{'image':'images/'+f'{split}_{i:05d}.png','mask':'masks/'+f'{split}_{i:05d}.png','split':split}
                                    for split,count in training['split_images'].items() for i in range(count)]}
    check(hashlib.sha256((json.dumps(historical_manifest,indent=2)+'\n').encode()).hexdigest()==training['manifest_sha256'],
          'procedural training manifest does not reproduce')
    protected=set()
    protected_count=0
    for offset,(split,count) in enumerate(training['split_images'].items()):
        for image,_ in generate(count,training['seed']+offset,'standard'):
            protected.add(hashlib.sha256(image.tobytes()).hexdigest())
            protected_count+=1
    for seed,case in zip(protocol['calibration']['seeds'],('standard','dim')):
        for image,_ in generate(protocol['calibration']['images']//2,seed,case):
            protected.add(hashlib.sha256(image.tobytes()).hexdigest())
            protected_count+=1
    seeds = {row['seed'] for row in report['results']}
    check(not seeds.intersection({training['seed'], training['seed']+1, training['seed']+2, *protocol['calibration']['seeds']}), 'training/calibration seed overlap')
    check(len(report['results'])==6 and {row['case'] for row in report['results']}==set(AUDIT_SEEDS), 'missing or duplicate image case')
    for row in report['results']:
        digest = hashlib.sha256()
        for image, labels in generate(row['images'], row['seed'], row['case']):
            check(hashlib.sha256(image.tobytes()).hexdigest() not in protected,
                  'audit image duplicates regenerated training/validation/legacy test/calibration image')
            digest.update(image.tobytes())
            digest.update(labels.tobytes())
        check(digest.hexdigest()==row['dataset_sha256'], 'image regeneration mismatch')
        check(row['images']==protocol['images_per_case'], 'image count mismatch')
        for method in ('cnn','hsv'):
            check(row[method]['latency_ms']['samples']==3*row['images'], 'timing samples mismatch')
            for values in row[method]['classes'].values():
                check(values['positive_images']+values['negative_images']==row['images'], 'class-image totals mismatch')
                check(values['detection_misses']<=values['positive_images'], 'miss denominator mismatch')
                check(values['false_positive_detections']<=values['negative_images'], 'false-positive denominator mismatch')
    return {'image_cases': len(report['results']), 'audit_images': sum(row['images'] for row in report['results']),
            'regenerated_protected_images':protected_count,'exact_audit_overlap':0,
            'selected_class_detector_calls': 6*sum(row['images'] for row in report['results'])}

def verify_trials(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    check(summary['protocol']==protocol, 'robot summary/protocol mismatch')
    for relative, digest in protocol['sha256'].items():
        check(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, 'robot source changed: '+relative)
    expected = set(itertools.product(protocol['cases'], protocol['targets'], protocol['seeds'], ('catalog','hsv')))
    rows = summary['trials']
    actual = [(r['case'],r['target'],r['seed'],r['backend']) for r in rows]
    check(len(actual)==len(set(actual)) and set(actual)==expected, 'missing or duplicate robot trial')
    samples = 0
    for row in rows:
        with gzip.open(directory/f"{row['trial_id']}.telemetry.jsonl.gz",'rt',encoding='utf-8') as stream:
            telemetry = [json.loads(line) for line in stream]
        check(all(p['target']==row['target'] for p in telemetry), 'wrong target telemetry')
        check(all(b['time']>a['time'] for a,b in zip(telemetry,telemetry[1:])), 'non-monotonic telemetry')
        check(all(b['time']-a['time']<=0.032*1.5 for a,b in zip(telemetry,telemetry[1:])), 'telemetry gaps')
        recomputed = score_trial(telemetry,row['case'],0.032,row['duration'])
        for key, value in recomputed.items():
            check(row[key]==value, 'scoring mismatch: '+row['trial_id']+'/'+key)
        check((directory/f"{row['trial_id']}.log").is_file(), 'missing process log')
        samples += len(telemetry)
    for case,target,seed in itertools.product(protocol['cases'],protocol['targets'],protocol['seeds']):
        pair = [r for r in rows if (r['case'],r['target'],r['seed'])==(case,target,seed)]
        check(pair[0]['initial_target']==pair[1]['initial_target'] and pair[0]['initial_robot_heading']==pair[1]['initial_robot_heading'], 'unmatched scene pair')
    return {'closed_loop_trials':len(rows), 'verified_telemetry_samples':samples,
            'matched_scene_pairs':len(rows)//2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',type=Path,default=ROOT/'reports/perception_audit')
    parser.add_argument('--trials',type=Path,default=ROOT/'reports/closed_loop_verified')
    parser.add_argument('--images-only',action='store_true')
    args=parser.parse_args()
    result = verify_images(args.images)
    if not args.images_only:
        result.update(verify_trials(args.trials))
        identity_path=args.trials/'identity_review.json'
        if identity_path.exists():
            from scripts.check_arrival_identity import compute
            check(json.loads(identity_path.read_text())==compute(args.trials), 'identity review does not reproduce')
            result['identity_review_verified']=True
    print(json.dumps({'verified':True,**result},indent=2))

if __name__=='__main__':
    main()
