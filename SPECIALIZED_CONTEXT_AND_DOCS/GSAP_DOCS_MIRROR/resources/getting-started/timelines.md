# Timelines

Just like we've just seen with staggers, It's common to animate more than one thing. But what if we need more control over the order and timing of those animations?

A lot of people reach for delays, *and they're not wrong*, delays do give us rudimentary control:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/eYMWJvP?default-tab=result\&theme-id=41164)

But this method of sequencing animations is a little fragile. What happens if we lengthen the duration of the first tween? The second and third tweens have **no awareness** of this change, so now there's an overlap - we'd have to increase all of the delays to keep them synchronized. If you've animated with CSS you will have run into this problem before!

Frustrating? Yep. That's why we created timelines!

## Timelines[​](#timelines "Direct link to Timelines")

info

Timelines are the key to creating easily adjustable, resilient sequences of animations. When you add tweens to a timeline, by default they'll play one-after-another in the order they were added.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/vYRmKKv?default-tab=result\&theme-id=41164)

```
// create a timeline
let tl = gsap.timeline()

// add the tweens to the timeline - Note we're using tl.to not gsap.to
tl.to(".green", { x: 600, duration: 2 });
tl.to(".purple", { x: 600, duration: 1 });
tl.to(".orange", { x: 600, duration: 1 });
```

But what if we want to add a gap or delay in between some of the tweens? One option would be to add a delay to a tween to offset it 's start time. But this isn't ***hugely*** flexible. What if we want tweens to overlap or start at the same time?

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ExEmGwr?default-tab=result\&theme-id=41164)

```
let tl = gsap.timeline()

tl.to(".green", { x: 600, duration: 2 });
tl.to(".purple", { x: 600, duration: 1, delay: 1 });
tl.to(".orange", { x: 600, duration: 1 });
```

## Position Parameter[​](#position-parameter "Direct link to Position Parameter")

This handy little parameter is the secret to building gorgeous sequences with precise timing. There are a variety of position parameters that we can use to position tweens pretty much anywhere! Take a look at this timeline...

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/VwXbRxq?default-tab=result\&theme-id=41164)

```
let tl = gsap.timeline()

// start at exactly 1 second into the timeline (absolute)
tl.to(".green", { x: 600, duration: 2 }, 1);

// insert at the start of the previous animation
tl.to(".purple", { x: 600, duration: 1 }, "<");

// insert 1 second after the end of the timeline (a gap)
tl.to(".orange", { x: 600, duration: 1 }, "+=1");
```

The most commonly used position parameters are the following -

1. **Absolute time** (in seconds) measured from the start of the timeline.

```
// insert exactly 3 seconds from the start of the timeline
tl.to(".class", {x: 100}, 3);
```

2. **A Gap**

```
//  1 second after the end of the timeline (usually the previously inserted animation)
tl.to(".class", {x: 100}, "+=1");
// beyond the end of the timeline by 50% of the inserting animation's total duration
tl.to(".class", {x: 100}, "+=50%");
```

3. **An Overlap**

```
//  1 second before the end of the timeline (this is usually the previously inserted animation)
tl.to(".class", {x: 100}, "-=1");

//  overlap with the end of the timeline by 25% of the inserting animation's total duration
tl.to(".class", {x: 100}, "-=25%");
```

Challenge

Position box **a** 3 seconds into the timeline.

Position box **b** 2 seconds after the end of box **a**.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/BavNmNm?default-tab=js%2Cresult\&editable=true\&theme-id=41164)

Deep Dive

Want to know more about the position parameter? [This article](/resources/position-parameter.md) covers it in more detail

## Special Properties[​](#special-properties "Direct link to Special Properties")

Timelines share most of the same special properties that tweens have like `repeat` and `delay` which allow you to control the entire sequence of animations as a whole!

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/NWYjZqQ?default-tab=js%2Cresult\&editable=true\&theme-id=41164)

## Timeline Defaults[​](#timeline-defaults "Direct link to Timeline Defaults")

tip

If you find yourself typing out a property over and over again, it might be time for `defaults`. Any property added to the defaults object in a timeline will be inherited by all the children that are created with the convenience methods like [to()](/docs/v3/GSAP/Timeline/to\(\).md), [from()](/docs/v3/GSAP/Timeline/from\(\).md), and [fromTo()](/docs/v3/GSAP/Timeline/fromTo\(\).md). This is a great way to keep your code concise.

```
var tl = gsap.timeline({defaults: {duration: 1}});

//no more repetition of duration: 1!
tl.to(".green", {x: 200})
  .to(".purple", {x: 200, scale: 0.2})
  .to(".orange", {x: 200, scale: 2, y: 20});
```
