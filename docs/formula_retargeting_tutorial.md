# Formula Retargeting Tutorial

This tutorial explains the current formula-based Quest 21-landmark to Aero Hand 7D action mapping.

The implementation lives in:

```text
aero_quest/retargeting.py
```

The live teleop entry point is:

```text
scripts/teleop/quest_tcp_aero_teleop.py
```

## 1. Goal

Quest gives 21 hand landmarks:

```text
P[0]  = wrist
P[1]  = ThumbMetacarpal
P[2]  = ThumbProximal
P[3]  = ThumbDistal
P[4]  = ThumbTip
P[5]  = IndexProximal
P[6]  = IndexIntermediate
P[7]  = IndexDistal
P[8]  = IndexTip
P[9]  = MiddleProximal
P[10] = MiddleIntermediate
P[11] = MiddleDistal
P[12] = MiddleTip
P[13] = RingProximal
P[14] = RingIntermediate
P[15] = RingDistal
P[16] = RingTip
P[17] = LittleProximal
P[18] = LittleIntermediate
P[19] = LittleDistal
P[20] = LittleTip
```

We convert those landmarks into a normalized Aero Hand action:

```text
a[0] = thumb_abduction
a[1] = thumb_flexion_1
a[2] = thumb_flexion_2
a[3] = index_curl
a[4] = middle_curl
a[5] = ring_curl
a[6] = little_curl
```

Every action channel is clamped to:

```text
0 <= a[i] <= 1
```

For curl channels:

```text
0 = open
1 = closed
```

For thumb abduction:

```text
0 = thumb inward/adducted
1 = thumb outward/abducted
```

## 2. Palm-Local Frame

Raw Quest landmarks are in world/device coordinates. We first convert them to a palm-local frame so the formula depends on hand shape, not global hand position.

Let:

```text
origin = P[0]
```

The palm x-axis points from little side to index side:

```text
x_axis = normalize(P[5] - P[17])
```

The middle-finger direction is used as a y hint:

```text
y_hint = normalize(P[9] - P[0])
```

The z-axis is perpendicular to the palm:

```text
z_axis = normalize(cross(x_axis, y_hint))
```

Then recompute y so the frame is orthogonal:

```text
y_axis = normalize(cross(z_axis, x_axis))
```

Stack axes into a rotation matrix:

```text
R = [x_axis, y_axis, z_axis]
```

Use wrist-to-middle-proximal distance as hand scale:

```text
scale = ||P[9] - P[0]||
```

The palm-local points are:

```text
P_local = ((P - origin) @ R) / scale
```

This removes translation, global rotation, and rough hand size differences.

## 3. Joint Bend Formula

For three points around a joint:

```text
A -- B -- C
```

the angle at `B` is:

```text
theta = arccos(
    dot(A - B, C - B) / (||A - B|| * ||C - B||)
)
```

When the finger is straight, `theta` is close to pi. When the finger bends, `theta` gets smaller.

So the bend amount is:

```text
bend = pi - theta
```

In code this is:

```python
joint_bend(points, a, b, c)
```

## 4. Bend Normalization

Raw bend angles are in radians. We map them to `[0, 1]`:

```text
normalized_bend = clamp(
    (bend - open_angle) / (closed_angle - open_angle),
    0,
    1
)
```

Current defaults:

```text
open_angle   = 0.08 rad
closed_angle = 1.35 rad
```

If fingers close too easily, increase `closed_angle`.

If fingers do not close enough, decrease `closed_angle`.

## 5. Four Finger Curl

For index/middle/ring/little, each finger has a proximal bend and distal bend.

Example for index:

```text
index_proximal = bend(P[5], P[6], P[7])
index_distal   = bend(P[6], P[7], P[8])
```

The final curl is:

```text
index_curl = normalize_bend(
    0.65 * index_proximal + 0.35 * index_distal
)
```

The same formula is used for middle, ring, and little:

```text
finger_curl = normalize_bend(
    0.65 * proximal_bend + 0.35 * distal_bend
)
```

The proximal joint has more weight because it visually dominates the hand closing motion.

## 6. Thumb Flexion

The thumb uses three bend measurements:

```text
thumb_base = bend(P[0], P[1], P[2])
thumb_mid  = bend(P[1], P[2], P[3])
thumb_tip  = bend(P[2], P[3], P[4])
```

Then:

```text
thumb_flexion_1 = normalize_bend(
    0.75 * thumb_base + 0.25 * thumb_mid
)
```

and:

```text
thumb_flexion_2 = normalize_bend(
    0.35 * thumb_mid + 0.65 * thumb_tip
)
```

So:

```text
a[1] = thumb_flexion_1
a[2] = thumb_flexion_2
```

## 7. Thumb Abduction / Adduction

This is the most delicate part.

A bad formula is:

```text
||ThumbTip - IndexProximal||
```

because plain thumb flexion also moves the thumb tip, making flexion look like inward adduction.

The current formula uses only the thumb base/proximal region:

```text
base_lateral = P_local[2].x - P_local[1].x
proximal_gap = P_local[2].x - P_local[5].x
```

Then:

```text
base_score = (base_lateral - 0.08) / (0.22 - 0.08)
gap_score  = (proximal_gap - 0.00) / (0.30 - 0.00)
```

Finally:

```text
thumb_abduction = clamp(
    0.75 * base_score + 0.25 * gap_score,
    0,
    1
)
```

This means:

```text
a[0] near 1 -> thumb is spread outward
a[0] near 0 -> thumb is pulled inward
```

Because it looks at `ThumbMetacarpal -> ThumbProximal`, plain thumb tip bending should not accidentally trigger inward adduction.

## 8. Full 7D Action Formula

Putting it together:

```text
a = [
    thumb_abduction,
    normalize_bend(0.75 * thumb_base + 0.25 * thumb_mid),
    normalize_bend(0.35 * thumb_mid  + 0.65 * thumb_tip),
    normalize_bend(0.65 * index_proximal  + 0.35 * index_distal),
    normalize_bend(0.65 * middle_proximal + 0.35 * middle_distal),
    normalize_bend(0.65 * ring_proximal   + 0.35 * ring_distal),
    normalize_bend(0.65 * little_proximal + 0.35 * little_distal),
]
```

Then:

```text
a = clamp(a, 0, 1)
```

## 9. Smoothing

Raw tracking can be noisy. We use exponential smoothing:

```text
a_filtered[t] = (1 - alpha) * a_filtered[t - 1] + alpha * a_raw[t]
```

Default:

```text
alpha = 0.25
```

Larger `alpha`:

```text
more responsive, more jitter
```

Smaller `alpha`:

```text
smoother, more lag
```

## 10. Timeout Safety

If no new Quest frame arrives for `timeout` seconds, the hand moves back toward a safe open action:

```text
safe_open_action = [1, 0, 0, 0, 0, 0, 0]
```

This means:

```text
thumb outward, all fingers open
```

The same smoothing formula is used:

```text
a_filtered = (1 - alpha) * a_filtered + alpha * safe_open_action
```

## 11. MuJoCo Ctrl Mapping

The Aero XML actuator order is not the semantic 7D action order.

The model reports:

```text
0 right_index_A_tendon
1 right_middle_A_tendon
2 right_ring_A_tendon
3 right_pinky_A_tendon
4 right_thumb_A_cmc_abd
5 right_th1_A_tendon
6 right_th2_A_tendon
```

So the code maps by actuator name, not by index.

Each normalized action value is converted to a MuJoCo ctrl value:

```text
ctrl = lo + value * (hi - lo)
```

For this hand, the channels are inverted:

```text
value_for_ctrl = 1 - action
```

So:

```text
curl action = 0 -> high tendon ctrl -> open
curl action = 1 -> low tendon ctrl  -> closed
```

## 12. How To Run

Start USB TCP forwarding:

```bash
adb devices
adb reverse tcp:8000 tcp:8000
adb reverse --list
```

Quest HTS App:

```text
Protocol: TCP
Target IP: localhost
Port: 8000
```

Run teleop:

```bash
python scripts/teleop/quest_tcp_aero_teleop.py --alpha 0.25
```

Run synthetic formula tests:

```bash
python tests/test_formula_retargeting.py
```

Expected test behavior:

```text
open hand:
  finger curls near 0

thumb flex only:
  thumb flexion increases
  thumb abduction stays high

fist:
  finger curls high
  thumb abduction lower

index only:
  index curl high
  other finger curls low
```

## 13. Tuning Guide

If fingers close too easily:

```text
increase closed_angle in normalize_bend()
```

If fingers do not close enough:

```text
decrease closed_angle in normalize_bend()
```

If thumb inward motion is too sensitive:

```text
increase the lower/upper thresholds in thumb_abduction_from_local()
```

If plain thumb bending still causes adduction:

```text
reduce the weight of proximal_gap
or use only base_lateral
```

Current formula:

```text
0.75 * base_score + 0.25 * gap_score
```

A more isolated version would be:

```text
thumb_abduction = clamp(base_score, 0, 1)
```
