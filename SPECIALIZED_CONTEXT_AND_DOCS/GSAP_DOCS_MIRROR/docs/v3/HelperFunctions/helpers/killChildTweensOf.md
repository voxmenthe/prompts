# Kill all tweens applied to child elements of a given target

You can kill all tweening elements that are children of a given target by using this function:

## Demo[​](#demo "Direct link to Demo")

```
function killChildTweensOf(ancestor) {
  ancestor = gsap.utils.toArray(ancestor)[0];
  gsap.globalTimeline
    .getChildren(true, true, false)
    .forEach((tween) =>
      tween
        .targets()
        .forEach((e) => e.nodeType && ancestor.contains(e) && tween.kill(e))
    );
}
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/yLOPRQG?default-tab=result\&theme-id=41164)
