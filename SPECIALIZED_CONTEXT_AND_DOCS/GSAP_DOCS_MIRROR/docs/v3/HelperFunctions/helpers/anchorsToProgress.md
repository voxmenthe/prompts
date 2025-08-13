# Calculate progress values for anchor points along a path

Calculate all the progress values for the anchor points on a path so that, for example, you could use DrawSVG to animate point-by-point (requires [MotionPathPlugin](/docs/v3/Plugins/MotionPathPlugin.md)):

```
// returns an array with the progress value (between 0 and 1) for each anchor along the path
function anchorsToProgress(rawPath, resolution) {
  resolution = ~~resolution || 12;
  if (!Array.isArray(rawPath)) {
    rawPath = MotionPathPlugin.getRawPath(rawPath);
  }
  MotionPathPlugin.cacheRawPathMeasurements(rawPath, resolution);
  let progress = [0],
    length,
    s,
    i,
    e,
    segment,
    samples;
  for (s = 0; s < rawPath.length; s++) {
    segment = rawPath[s];
    samples = segment.samples;
    e = segment.length - 6;
    for (i = 0; i < e; i += 6) {
      length = samples[(i / 6 + 1) * resolution - 1];
      progress.push(length / rawPath.totalLength);
    }
  }
  return progress;
}
```

## Demo[​](#demo "Direct link to Demo")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/mdyxvGX?default-tab=result\&theme-id=41164)
