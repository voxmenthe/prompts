# scrollTrigger

### scrollTrigger: [ScrollTrigger](/docs/v3/Plugins/ScrollTrigger/.md) | undefined

A handy way to access the ScrollTrigger associated with a timeline. This is only accessible if the timeline has a ScrollTrigger.

### Details[​](#details "Direct link to Details")

warning

A `scrollTrigger`property is only added to the Timeline or Tween *if* it has a ScrollTrigger.

See the [ScrollTrigger](/docs/v3/Plugins/ScrollTrigger/.md) docs for more details

```
// add a ScrollTrigger to a Timeline
let tl = gsap.timeline({scrollTrigger: {start: "top center"...}});

// access the ScrollTrigger to call various methods
tl.scrollTrigger.refresh();
// or
tl.scrollTrigger.kill();
```
