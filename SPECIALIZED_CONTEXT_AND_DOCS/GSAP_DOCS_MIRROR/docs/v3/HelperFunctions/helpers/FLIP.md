# Flip

If you're shifting things around in the DOM and then you want elements to animate to their new positions, the most full-featured way to handle it is with the [Flip Plugin](/docs/v3/Plugins/Flip/.md), but if you're only doing basic things you can use this helper function (see the comments at the top to learn how to use it):

```
/*
Copy this to your project. Pass in the elements (selector text or NodeList or Array), then a
function/callback that actually performs your DOM changes, and optionally a vars
object that contains any of the following properties to customize the transition:

- duration [Number] - duration (in seconds) of each animation
- stagger [Number | Object | Function] - amount to stagger the starting time of each animation. You may use advanced staggers too (see https://codepen.io/GreenSock/pen/jdawKx)
- ease [Ease] - controls the easing of the animation. Like "power2.inOut", or "elastic", etc.
- onComplete [Function] - a callback function that should be called when all the animation has completed.
- delay [Number] - time (in seconds) that should elapse before any of the animations begin.

This function will return a Timeline containing all the animations.
*/
function flip(elements, changeFunc, vars) {
  elements = gsap.utils.toArray(elements);
  vars = vars || {};
  let tl = gsap.timeline({
      onComplete: vars.onComplete,
      delay: vars.delay || 0,
    }),
    bounds = elements.map((el) => el.getBoundingClientRect()),
    copy = {},
    p;
  elements.forEach((el) => {
    el._flip && el._flip.progress(1);
    el._flip = tl;
  });
  changeFunc();
  for (p in vars) {
    p !== "onComplete" && p !== "delay" && (copy[p] = vars[p]);
  }
  copy.x = (i, element) =>
    "+=" + (bounds[i].left - element.getBoundingClientRect().left);
  copy.y = (i, element) =>
    "+=" + (bounds[i].top - element.getBoundingClientRect().top);
  return tl.from(elements, copy);
}
```

**DEMO**

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/eYJLOdj?default-tab=result\&theme-id=41164)
