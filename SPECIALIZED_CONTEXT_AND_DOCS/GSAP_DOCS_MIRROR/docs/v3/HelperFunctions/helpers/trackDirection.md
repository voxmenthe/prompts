# trackDirection

## Need an onReverse()? Track the playhead direction of any animation[​](#need-an-onreverse-track-the-playhead-direction-of-any-animation "Direct link to Need an onReverse()? Track the playhead direction of any animation")

If you find yourself needing an onReverse() callback (which doesn't exist) or a way to get notified when the playhead changes direction, this is a very useful helper function. What makes it special is that it works no matter how deeply-nested the animation is. Remember, the parent or parent's parent could get reversed or a negative timeScale which directly affects how the playhead sweeps across the descendants.

```
function trackDirection(value) {
  typeof value !== "object" && (value = { onUpdate: value });
  let prevTime = 0,
    prevReversed = false,
    anim = value.eventCallback ? value : value.animation,
    onUpdate = value.onUpdate,
    onToggle = value.onToggle;
  return anim
    ? anim.eventCallback(
        "onUpdate",
        trackDirection({ onUpdate: onUpdate, onToggle: onToggle })
      )
    : function () {
        let time = this.totalTime(),
          reversed = time < prevTime;
        this.direction = reversed ? -1 : 1;
        if (reversed !== prevReversed) {
          onToggle && onToggle.call(this, this.direction);
          prevReversed = reversed;
        }
        prevTime = time;
        onUpdate && onUpdate.call(this, this.direction);
      };
}
```

## Usage[​](#usage "Direct link to Usage")

Choose from any of the following:

1. Directly as a callback (it returns a function):

   <!-- -->

   ```
   gsap.to(... {onUpdate: trackDirection(), ...})
   ```

2. Assigned to the animation:

   <!-- -->

   ```
   let tl = gsap.timeline();
   trackDirection(tl);
   ```

You can add configuration options (onToggle and/or onUpdate):

```
gsap.to(
  ...{
    x: 100,
    onUpdate: trackDirection({
      onToggle: (direction) => console.log("toggled direction to", direction),
      onUpdate: (direction) => console.log("updated animation"),
    }),
  }
);
```

Or when assigned to the animation:

```
trackDirection({
  animation: tl,
  onToggle: (direction) => console.log("toggled direction to", direction),
  onUpdate: (direction) => console.log("updated animation"),
});
```

warning

since "direction" is set whenever the playhead changes position, it won't update immediately. For example, if you call tween.reverse() and then immediately check (before the next tick), tween.direction will still report as 1 because the playhead hasn't moved yet.

## Demo[​](#demo "Direct link to Demo")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/rNyrGjB?default-tab=result\&theme-id=41164)
