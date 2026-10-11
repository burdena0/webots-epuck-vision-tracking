"""Scoring helpers used only by the supervisor and offline report."""
import math
import statistics

def circle_box_gap(position, center, half_size=(0.06, 0.06), radius=0.037):
    return math.hypot(max(abs(position[0]-center[0])-half_size[0], 0),
                      max(abs(position[1]-center[1])-half_size[1], 0)) - radius

def score_trial(samples, case, timestep_seconds, required_duration):
    if not samples or samples[-1]['time'] < required_duration-timestep_seconds*1.5:
        raise ValueError('incomplete telemetry cannot count as a trial')
    if any(not all(math.isfinite(p[k]) for k in ('time', 'left', 'right', 'inference_ms')) for p in samples):
        raise ValueError('non-finite telemetry cannot be scored')
    if any(p['state'] in ('sensor_fault', 'vision_fault') for p in samples):
        raise ValueError('controller faults invalidate the trial')
    collisions = sum(bool(p['contacts']) for p in samples)
    gaps = [p['target_gap_m'] for p in samples if p['target_visible']]
    min_gap = min(gaps) if gaps else None
    # Contacts include actual e-puck descendants against target/distractors only.
    # The geometric check additionally flags approximate overlap beyond 1 mm.
    collision = collisions > 0 or (min_gap is not None and min_gap < -0.001)
    stable_started = None
    arrival = None
    for p in samples:
        stable = (p['target_visible'] and p.get('target_nearest', True) and 0 <= p['target_gap_m'] <= 0.08 and
                  p['state'] == 'blocked' and abs(p['left']) <= 1e-6 and abs(p['right']) <= 1e-6)
        if stable:
            if stable_started is None:
                stable_started = p['time']
            if p['time']-stable_started >= 0.5 and arrival is None:
                arrival = stable_started
        else:
            stable_started = None
    forward = [max(0, (p['left']+p['right'])/2) for p in samples]
    hidden = [max(0, (p['left']+p['right'])/2) for p in samples if 3.15 <= p['time'] < 5.9 and case == 'occlusion']
    # Exclude the first 30 perception samples from steady-state timing summaries.
    latency = [p['inference_ms'] for p in samples[30:]]
    success = not collision and ((max(forward) <= 1e-6 and arrival is None) if case == 'missing' else arrival is not None)
    return {'success': success, 'arrival_sim_seconds': arrival, 'collision': collision,
            'object_contact_samples': collisions, 'minimum_target_gap_m': min_gap,
            'max_forward_rad_s': max(forward), 'max_forward_during_occlusion_rad_s': max(hidden) if hidden else None,
            'inference_median_ms': statistics.median(latency) if latency else None,
            'inference_p95_ms': sorted(latency)[math.ceil(0.95*len(latency))-1] if latency else None,
            'samples': len(samples), 'states': sorted({p['state'] for p in samples})}

def wilson(successes, count):
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError('invalid binomial counts')
    z = 1.959963984540054
    p = successes/count
    denom = 1+z*z/count
    center = (p+z*z/(2*count))/denom
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denom
    return [max(0, center-half), min(1, center+half)]
