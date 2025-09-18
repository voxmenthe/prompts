# Control and Callbacks

## Control Methods[​](#control-methods "Direct link to Control Methods")

All the animations we 've looked at so far play on page load or after a `delay`. But what if you want a little more control over your animation? A common use case is to play an animation on a certain interaction like a button click or hover.

Control methods can be used on both tweens and timelines and allow you to [play](/docs/v3/GSAP/Tween/play\(\).md), [pause](/docs/v3/GSAP/Tween/pause\(\).md), [reverse](/docs/v3/GSAP/Tween/reverse\(\).md) or even [speed up](/docs/v3/GSAP/Tween/timeScale\(\).md) your animations!

```
// store the tween or timeline in a variable
let tween = gsap.to("#logo", {duration: 1, x: 100});

//pause
tween.pause();

//resume (honors direction - reversed or not)
tween.resume();

//reverse (always goes back towards the beginning)
tween.reverse();

//jump to exactly 0.5 seconds into the tween
tween.seek(0.5);

//jump to exacty 1/4th into the tween 's progress:
tween.progress(0.25);

//make the tween go half-speed
tween.timeScale(0.5);

//make the tween go double-speed
tween.timeScale(2);

//immediately kill the tween and make it eligible for garbage collection
tween.kill();

// You can even chain control methods
// Play the timeline at double speed - in reverse.
tween.timeScale(2).reverse();
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/OJLgdyg?default-tab=result\&theme-id=41164)

tip

Clients love to make last minute tweaks to animation! **[timeScale](/docs/v3/GSAP/Timeline/timeScale\(\).md)** comes in really handy for speeding up or slowing down complex animation timelines without having to change lots of durations and delays.

## Callbacks[​](#callbacks "Direct link to Callbacks")

If you need to know when an animation starts, or maybe run some JS when an animation comes to an end, you can use **Callbacks**. All tweens and timelines have these callbacks:

* **onComplete**: invoked when the animation has completed.
* **onStart**: invoked when the animation begins
* **onUpdate**: invoked every time the animation updates (on every frame while the animation is active).
* **onRepeat**: invoked each time the animation repeats.
* **onReverseComplete**: invoked when the animation has reached its beginning again when reversed.

```
gsap.to(".class", {
  duration: 1, 
  x: 100, 
  // arrow functions are handy for concise callbacks
  onComplete: () => console.log("the tween is complete")
});

// If your function doesn't fit neatly on one line, no worries.
// you can write a regular function and reference it
gsap.timeline({onComplete: tlComplete}); // <- no () after the reference!

function tlComplete() {
  console.log("the tl is complete");
  // more code
}
```

Use case: interaction events that trigger animations

Inside of event listeners for user interaction events, we can use control methods to have fine control over our animation’s play state.

In the example below, we are creating a timeline for each element (so that it doesn’t fire the same animation on all instances), attaching a reference for that timeline to the element itself, and then playing the relevant timeline when the element is hovered, reversing it when the mouse leaves. We're also adjusting the speed so it's faster on reverse and slower on entry. This is a good UX pattern.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/PoqRZOB?default-tab=result\&theme-id=41164)
