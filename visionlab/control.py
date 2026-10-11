from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Command:
    left: float
    right: float
    state: str
    error: float | None

class Policy:
    def __init__(self, max_speed=2.2, gain=3.0, acceleration=6.0):
        self.max_speed, self.gain, self.acceleration = max_speed, gain, acceleration
        self.filtered_x = None
        self.lost_time = 0.0
        self.blocked = False
        self.last = (0.0, 0.0)

    def halt(self, state):
        # safety stops skip the speed ramp
        self.last = (0.0, 0.0)
        self.filtered_x = None
        return Command(0.0, 0.0, state, None)

    def step(self, detection, proximity, dt):
        if len(proximity) != 8 or not math.isfinite(dt) or dt <= 0 or not all(math.isfinite(x) for x in proximity):
            return self.halt('sensor_fault')
        front = max(proximity[i] for i in (0, 1, 6, 7))
        self.blocked = front >= (100 if self.blocked else 130)
        if self.blocked:
            return self.halt('blocked')
        if detection is not None and not math.isfinite(detection.confidence):
            return self.halt('vision_fault')
        if detection is None or detection.confidence < 0.65:
            self.lost_time += dt
            if self.lost_time < 0.4 or self.lost_time > 5.0:
                return self.halt('lost')
            # search in place for a bit. dont drive blind
            direction = 1 if int((self.lost_time - 0.4) / 0.65) % 2 == 0 else -1
            desired, state, error = (0.65 * direction, -0.65 * direction), 'search', None
        else:
            self.lost_time = 0.0
            if not math.isfinite(detection.x) or not 0 <= detection.x <= 1:
                return self.halt('vision_fault')
            alpha = 1.0 - math.exp(-dt / 0.10)
            self.filtered_x = detection.x if self.filtered_x is None else self.filtered_x + alpha * (detection.x - self.filtered_x)
            error = 2 * self.filtered_x - 1
            turn = self.gain * error if abs(error) > 0.025 else 0.0
            forward = self.max_speed * max(0.0, 1 - abs(error) / 0.65)
            forward *= max(0.15, min(1.0, (130 - front) / 50))
            desired = (forward + turn, forward - turn)
            state = 'track' if abs(error) > 0.08 else 'approach'
        desired = tuple(max(-3.0, min(3.0, x)) for x in desired)
        ramp = self.acceleration * dt
        wheels = tuple(old + max(-ramp, min(ramp, new - old)) for old, new in zip(self.last, desired))
        self.last = wheels
        return Command(*wheels, state, error)

class OriginalPolicy:
    def step(self, detection, proximity, dt):
        if detection is None:
            return Command(0, 0, 'lost', None)
        error = (detection.x - 0.5) * 320
        if abs(error) > 10:
            turn = 1 if error > 0 else -1
            return Command(turn, -turn, 'track', error / 160)
        if max(proximity[0], proximity[7]) > 180:
            return Command(0, 0, 'blocked', error / 160)
        return Command(0.5, 0.5, 'approach', error / 160)
