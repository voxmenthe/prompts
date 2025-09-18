# Keyframes

info

keyframes are only to be used in `to()` tweens.

If you find yourself writing multiple tweens to animate one target, it ***may*** be time to reach for keyframes. Keyframes are a great way to move a target through a series of steps while keeping your code concise.

Take a repetitive timeline like the one below - It can be simplified down nicely to fit into one tween:

```
// timeline
let tl = gsap.timeline();
tl.to(".box", {
    x: 100
  })
  .to(".box", {
    y: 100
  })
  .to(".box", {
    x: 0
  })
  .to(".box", {
    y: 0
  });

// Array-based keyframes
gsap.to(".box", {
  keyframes: {
    x: [0, 100, 100, 0, 0],
    y: [0, 0, 100, 100, 0],
    ease: "power1.inOut"
  },
  duration: 2
});
```

**We like to think of keyframes as a sub-timeline nested *inside* a tween.** There are a few different ways to write keyframes. If you're a visual learner, check out this video.

Video Walkthrough

## Keyframe Options[​](#keyframe-options "Direct link to Keyframe Options")

### Object keyframes - v3.0[​](#object-keyframes---v30 "Direct link to Object keyframes - v3.0")

This keyframes syntax lets you pass in an Array of vars parameters to use for the given target(s). Think of them like a sequence of .to() tween vars. You can use a `delay` value to create gaps or overlaps.

The default per-keyframe ease is `linear` which you can override in individual keyframes. You can also apply an ease to the *entire* keyframe sequence.

```
gsap.to(".elem", {
 keyframes: [
  {x: 100, duration: 1, ease: 'sine.out'}, // finetune with individual eases
  {y: 200, duration: 1, delay: 0.5}, // create a 0.5 second gap
  {rotation: 360, duration: 2, delay: -0.25} // overlap by 0.25 seconds
 ],
 ease: 'expo.inOut' // ease the entire keyframe block
});
```

### Percentage keyframes - v3.9[​](#percentage-keyframes---v39 "Direct link to Percentage keyframes - v3.9")

This familiar syntax makes porting animations over from CSS a breeze! Instead of using delays and duration in the keyframe object, you specify an overall duration on the tween itself, then define the position of each keyframe using percentages.

To be consistent with CSS behaviour, the default per-keyframe ease is `power1.inOut` which generally looks quite nice but you can override this in individual keyframes or on all keyframes using `easeEach`.

```
gsap.to(".elem", {
 keyframes: {
  "0%":   { x: 100, y: 100},
  "75%":  { x: 0, y: 0, ease: 'sine.out'}, // finetune with individual eases
  "100%": { x: 50, y: 50 },
   easeEach: 'expo.inOut' // ease between keyframes
 },
 ease: 'none' // ease the entire keyframe block
 duration: 2,
})
```

### Simple Array-based keyframes - v3.9[​](#simple-array-based-keyframes---v39 "Direct link to Simple Array-based keyframes - v3.9")

Just define an Array of values and they'll get equally distributed over the time specified in the tween.

The default per-keyframe ease is `power1.inOut`, but you can override this by using `easeEach`. The Arrays do not need to have the same number of elements.

```
gsap.to(".elem", {
 keyframes: {
  x: [100, 0, 50],
  y: [100, 0, 50]
  easeEach: 'sine.inOut' // ease between keyframes
  ease: 'expo.out' // ease the entire keyframe block
 },
 duration: 2,
})
```

## Easing keyframes[​](#easing-keyframes "Direct link to Easing keyframes")

Easing is integral to animation and keyframes give you a huge amount of flexibility.

**Percentage keyframes** and **Simple keyframes** allow you to control the ease between each of the keyframes with `easeEach`.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/941b82d684b7fbf5303304d671e15ce2?default-tab=result\&theme-id=41164)

With **Object keyframes** and **Percentage keyframes** you can drill down and add different eases into individual keyframes.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/1c38f65c7536803a59f19a899ac0fbb9?default-tab=result\&theme-id=41164)

You can even combine multiple easing properties, keyframes and normal tween values. 🤯

```
gsap.to(".box", {
  keyframes: {
    y: [0, 80, -10, 30, 0],
    ease: "none", // <- ease across the entire set of keyframes (defaults to the one defined in the tween, or "none" if one isn't defined there)
    easeEach: "power2.inOut" // <- ease between each keyframe (defaults to "power1.inOut")
  },
  rotate: 180,
  ease: "elastic", // <- the "normal" part of the tween. In this case, it affects "rotate" because it's outside the keyframes
  duration: 5,
  stagger: 0.2
});
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/92533379938b4afb117727d84b47651b?default-tab=result\&theme-id=41164)

### Keyframe tips[​](#keyframe-tips "Direct link to Keyframe tips")

Both the **Object keyframes** and the **Percentage keyframes** behave similarly to tweens, so you can leverage callbacks like `onStart` and `onComplete`.

```
gsap.to(".elem", {
 keyframes: [
  {x: 100, duration: 1},
  {y: 200, duration: 1, onComplete: () => { console.log('complete')}},
  {rotation: 360, duration: 2, delay: -0.25, ease: 'sine.out'}
 ]
});

gsap.to(".elem", {
 keyframes: {
  "0%":   { x: 100, y: 100},
  "75%":  { x: 0, y: 0, ease: 'power3.inOut'},
  "100%": { x: 50, y: 50, ease: 'none', onStart: () => { console.log('start')} }
 },
 duration: 2,
})
```

We hope this has helped you get your head around keyframes - if you have any questions pop over to [our forums](https://gsap.com/community/).

Happy tweening!
