"""The long jump's tank schedule (docs/JUMP.md section 7), for `--tank-script`.

Blow the tank while the legs push (the submarine kit goes from beta 0.86 to
1.02), hold the base 0.30 m above the ground with the tank (PI on height, D on
vertical speed, around the neutral air fraction 0.875), and flood at 8 s to sink
and land. With a current the robot is carried for the whole hover: 0.15 m/s took
it 1.8 m (native, 2026-10-27).
"""
NEUTRAL = 0.875          # beta ~1.0 with the submarine keel
HOVER_Z, KP, KI, KD = 0.30, 4.0, 1.5, 6.0
_i = [0.0, None]
def air(t, s):
    if t < 0.6:
        _i[0], _i[1] = 0.0, t
        return 1.0                      # blow while the legs push
    if t < 8.0:
        dt = t - _i[1] if _i[1] is not None else 0.0
        _i[1] = t
        _i[0] += (HOVER_Z - s["z"]) * dt
        return NEUTRAL + KP * (HOVER_Z - s["z"]) + KI * _i[0] - KD * s["vz"]
    return 0.0                          # flood: sink and land
