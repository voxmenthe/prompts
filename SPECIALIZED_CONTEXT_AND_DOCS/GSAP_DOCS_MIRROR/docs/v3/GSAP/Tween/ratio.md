# ratio

### ratio

**\[read-only]** the progress of the Tween (a value between 0 and 1 where 0.5 is in the middle) **after** being run through the `ease`. So this value may exceed the 0-1 range, like in the case of `ease: "back"` or `ease: "elastic"`. It can be useful as a multiplier for your own interpolation, like in an `onUpdate` callback.

### Details[​](#details "Direct link to Details")

\[read-only] the progress of the Tween (a value between 0 and 1 where 0.5 is in the middle) **after** being run through the `ease`. So this value may exceed the 0-1 range, like in the case of `ease: "back"` or `ease: "elastic"`. It can be useful as a multiplier for your own interpolation, like in an `onUpdate` callback.

So if you have a one second tween with an ease of `"power2.out"`, at the 0.5 second mark (where the progress is also half way), `tween.progress()` will report 0.5 while `tween.ratio` will report 0.875. As the code below shows, `this.ratio` is always equal to value you can obtain from passing the tween's `.progress()` into the ease function.

```
const easeFunc = gsap.parseEase("power2.out");
const tween = gsap.to({ foo: 0 }, { foo: 10, duration: 1, ease: "power2.out" });
tween.pause(0.5); // pause at 0.5 seconds which is halfway in this 1-second tween

console.log(tween.progress()); // 0.5
console.log(tween.ratio); // 0.875
console.log(easeFunc(tween.progress())); // 0.875
```
