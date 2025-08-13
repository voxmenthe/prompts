# SteppedEase

info

SteppedEase is included in GSAP's core

### Description[​](#description "Direct link to Description")

Most easing equations give a smooth, gradual transition between the start and end values, but SteppedEase provides an easy way to define a specific number of steps that the transition should take.

For example, if x is 0 and you want to tween it to 100 with 5 steps (20, 40, 60, 80, and 100) over the course of 2 seconds, you'd do:

```
gsap.to(obj, {duration: 2, x: 100, ease: "steps(5)"});
```

**Note:** SteppedEase is optimized for use with the GreenSock Animation Platform, so it isn't intended to be used with other engines. Specifically, its easing equation always returns values between 0 and 1.
