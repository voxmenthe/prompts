# "Compensated" skews

This is a special method that you can apply via an onUpdate to make a tween render skews in the old `skewType: "compensated"` way from GSAP 2. Note that it affects an element's scaleX/scaleY (hence "compensated")! This assumes skews are degree-based, and only works in GSAP 3. This is not an "officially supported" method.:

```
function compensatedSkew() {
  var targets = this.targets(),
    i = targets.length,
    DEG2RAD = Math.PI / 180,
    target,
    scaleY,
    scaleX,
    cache;
  while (i--) {
    target = targets[i];
    cache = target._gsap;
    scaleY = cache.scaleY;
    scaleX = cache.scaleX;
    cache.scaleY *= Math.cos(parseFloat(cache.skewX) * DEG2RAD);
    cache.scaleX *= Math.cos(parseFloat(cache.skewY) * DEG2RAD);
    cache.renderTransform(1, cache);
    cache.scaleY = scaleY;
    cache.scaleX = scaleX;
  }
}

// usage:
gsap.set(target, { skewX: -30, onUpdate: compensatedSkew });
```
