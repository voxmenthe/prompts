# Blend two eases

If you need one ease at the start of your animation, and a different one at the end, you can use this function to blend them!

```
//just feed in the starting ease and the ending ease (and optionally an ease to do &#xFEFF;the blending), and it'll return a new Ease that's...blended!
function blendEases(startEase, endEase, blender) {
  var parse = function (ease) {
      return typeof ease === "function" ? ease : gsap.parseEase("power4.inOut");
    },
    s = gsap.parseEase(startEase),
    e = gsap.parseEase(endEase),
    blender = parse(blender);
  return function (v) {
    var b = blender(v);
    return s(v) * (1 - b) + e(v) * b;
  };
}
//example usage:
gsap.to("#target", {
  duration: 2,
  x: 100,
  ease: blendEases("back.in(1.2)", "bounce"),
});
```

**DEMO**

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/59008f3ee5f26b811e90017bdcbb7dfb?default-tab=result\&theme-id=41164)

If you need to **invert** an ease instead, see [this demo](https://codepen.io/GreenSock/pen/rgROxY?editors=0010) for a different helper function.
