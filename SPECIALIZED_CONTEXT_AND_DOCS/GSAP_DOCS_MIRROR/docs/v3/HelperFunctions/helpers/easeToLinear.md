# Estimate where an ease will hit a certain value

An ease accepts a normalized progress value (0-1) and returns the corresponding eased value, but what if you want to know when that eased value will hit a specific ratio like 0.7? For example, a `"power2.out"` ease may hit 0.7 when the linear progress is only around 0.33. This function lets you feed in that 0.7 and get the linear progress value (0.33 in this example):

```
function easeToLinear(ease, ratio, precision = 0.0001) {
  ease = gsap.parseEase(ease);
  let t = 0,
    dif = ratio - ease(t),
    inc = dif / 2,
    newDif;
  while (Math.abs(dif) > precision) {
    newDif = ratio - ease((t += inc));
    newDif < 0 !== inc < 0 && (inc *= Math.max(-0.5, newDif / dif));
    dif = newDif;
  }
  return t + ((ratio - ease(t + inc)) / dif) * -inc;
}
```

More practical use: let's say you're animating a value from 100 to 500 with a `"power2.out"` ease and you want to estimate the linear progress value (between 0 and 1) where it'll hit 250 according to that ease - you could leverage this function like:

```
let from = 100,
  to = 500,
  targetValue = 250,
  progress = easeToLinear(
    "power2",
    (targetValue - from) / (to - from),
    0.00001
  );
```
