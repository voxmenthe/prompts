# Align transformOrigin of two elements smoothly

Instantly change the `transformOrigin` of one element to align with another element's `transformOrigin` without a jump (requires [MotionPathPlugin](/docs/v3/Plugins/MotionPathPlugin/static.convertCoordinates\(\).md)):

```
// fromElement is the one whose transformOrigin should change to match up with the toElement's transformOrigin.
function alignOrigins(fromElement, toElement) {
  let [fromEl, toEl] = gsap.utils.toArray([fromElement, toElement]),
    a = window.getComputedStyle(toEl).transformOrigin.split(" "),
    newOrigin = MotionPathPlugin.convertCoordinates(toEl, fromEl, {
      x: parseFloat(a[0]),
      y: parseFloat(a[1]),
    }),
    bounds1 = fromEl.getBoundingClientRect(),
    bounds2;
  gsap.set(fromEl, {

    transformOrigin: newOrigin.x + "px " + newOrigin.y + "px",
  });
  bounds2 = fromEl.getBoundingClientRect();
  gsap.set(fromEl, {
    x: "+=" + (bounds1.left - bounds2.left),
    y: "+=" + (bounds1.top - bounds2.top),
  });
}
```
