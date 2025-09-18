# Change transformOrigin without a jump

If you want to change `transformOrigin` dynamically without a jump, you'd need to compensate its translation (x/y). Here's a function I whipped together for that purpose:

```
function smoothOriginChange(targets, transformOrigin) {
  gsap.utils.toArray(targets).forEach(function (target) {
    var before = target.getBoundingClientRect();
    gsap.set(target, { transformOrigin: transformOrigin });
    var after = target.getBoundingClientRect();
    gsap.set(target, {
      x: "+=" + (before.left - after.left),
      y: "+=" + (before.top - after.top),
    });
  });
}
```

**DEMO**

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/feb24447fb7c6a82e777a0110c281a31?default-tab=result\&theme-id=41164)
