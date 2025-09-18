# The Position Parameter

The secret to building gorgeous sequences with precise timing is understanding the `position` parameter which is used in many methods throughout GSAP. This one super-flexible parameter controls the placement of your tweens, labels, callbacks, pauses, and even nested timelines, so you'll be able to literally place anything anywhere in any sequence.

info

For a quick overview of the [position parameter](/resources/position-parameter.md), check out this video from the ["GSAP 3 Express" course](https://courses.snorkl.tv/courses/gsap-3-express?ref=44f484) by Snorkl.tv - one of the best ways to learn the basics of GSAP.

## Using position with gsap.to()[​](#using-position-with-gsapto "Direct link to Using position with gsap.to()")

This article will focus on the [`gsap.to()`](/docs/v3/GSAP/gsap.to\(\).md) method for adding tweens to a Tween, but it works the same in other methods like [`from()`](/docs/v3/GSAP/gsap.from\(\).md), [`fromTo()`](/docs/v3/GSAP/gsap.fromTo\(\).md), [`add()`](/docs/v3/GSAP/Timeline/add\(\).md), etc. Notice that the `position` parameter comes after the `vars` parameter:

```
.to( target, vars, **position** )
```

Since it's so common to chain animations one-after-the-other, the default position is `"+=0"` which just means "at the end", so `timeline.to(...).to(...)` chains those animations back-to-back. It's fine to omit the [position parameter](/resources/position-parameter.md) in this case. But what if you want them to overlap, or start at the same time, or have a gap between them?

No problem.

## Multiple behaviors[​](#multiple-behaviors "Direct link to Multiple behaviors")

The [position parameter](/resources/position-parameter.md) is **super** flexible, accommodating any of these options:

* **Absolute time** (in seconds) measured from the start of the timeline, as a **number** like `3`

  ```
  // insert exactly 3 seconds from the start of the timeline
  tl.to(".class", {x: 100}, 3);
  ```

* **Label**, like `"someLabel"`. *If the label doesn't exist, it'll be added to the end of the timeline.*

  ```
  // insert at the "someLabel" label
  tl.to(".class", {x: 100}, "someLabel");
  ```

* `"<"` The **start** of previous animation\*\*. *Think of `<` as a pointer back to the start of the previous animation.*

  ```
  // insert at the START of the  previous animation
  tl.to(".class", {x: 100}, "<");
  ```

* `">"` - The **end** of the previous animation\*\*. *Think of `>` as a pointer to the end of the previous animation.*

  ```
  // insert at the END of the previous animation
  tl.to(".class", {x: 100}, ">");
  ```

* A complex string where `"+="` and `"-="` prefixes indicate **relative** values. *When a number follows `"<"` or `">"`, it is interpreted as relative so `"<2"` is the same as `"<+=2"`.* Examples:

  * `"+=1"` - 1 second past the end of the timeline (creates a gap)
  * `"-=1"` - 1 second before the end of the timeline (overlaps)
  * `"myLabel+=2"` - 2 seconds past the label `"myLabel"`
  * `"<+=3"` - 3 seconds past the start of the previous animation
  * `"<3"` - same as `"<+=3"` (see above) (`"+="` is implied when following `"<"` or `">"`)
  * `">-0.5"` - 0.5 seconds before the end of the previous animation. It's like saying *"the end of the previous animation plus -0.5"*

* A complex string based on a **percentage**. When immediately following a `"+="` or `"-="` prefix, the percentage is based on [total duration](/docs/v3/GSAP/Tween/totalDuration\(\).md) of the **animation being inserted**. When immediately following `"&lt"` or `">"`, it's based on the [total duration](/docs/v3/GSAP/Tween/totalDuration\(\).md) of the **previous animation**. *Note: total duration includes repeats/yoyos*. Examples:

  * `"-=25%"` - overlap with the end of the timeline by 25% of the inserting animation's total duration
  * `"+=50%"` - beyond the end of the timeline by 50% of the inserting animation's total duration, creating a gap
  * `"<25%"` - 25% into the previous animation (from its start). Same as `">-75%"` which is negative 75% from the **end** of the previous animation.
  * `"<+=25%"` - 25% of the inserting animation's total duration past the start of the previous animation. Different than `"<25%"` whose percentage is based on the **previous animation's** total duration whereas anything immediately following `"+="` or `"-="` is based on the **inserting animation's** total duration.
  * `"myLabel+=30%"` - 30% of the inserting animation's total duration past the label `"myLabel"`.

## Basic code usage[​](#basic-code-usage "Direct link to Basic code usage")

```
tl.to(element, 1, {x: 200})
  //1 second after end of timeline (gap)
  .to(element, {duration: 1, y: 200}, "+=1")
  //0.5 seconds before end of timeline (overlap)
  .to(element, {duration: 1, rotation: 360}, "-=0.5")
  //at exactly 6 seconds from the beginning of the timeline
  .to(element, {duration: 1, scale: 4}, 6);
```

It can also be used to add tweens at labels or relative to labels

```
//add a label named scene1 at an exact time of 2-seconds into the timeline
tl.add("scene1", 2)
  //add tween at scene1 label
  .to(element, {duration: 4, x: 200}, "scene1")
  //add tween 3 seconds after scene1 label
  .to(element, {duration: 1, opacity: 0}, "scene1+=3");
```

## Interactive Demos[​](#interactive-demos "Direct link to Interactive Demos")

Sometimes technical explanations and code snippets don't do these things justice. Take a look at the interactive examples below.

### No position: Direct Sequence[​](#no-position-direct-sequence "Direct link to No position: Direct Sequence")

If no [position parameter](/resources/position-parameter.md) is provided, all tweens will run in direct succession.

0

1

2blueSpin

3

4

Play

```
```

### Positive Relative: Gaps / Delays[​](#positive-relative-gaps--delays "Direct link to Positive Relative: Gaps / Delays")

Use a positive, relative value ("+=X") to place your tween X seconds after previous animations end

0

1

2blueSpin

3

4

Play

```
```

### Negative Relative: Overlap[​](#negative-relative-overlap "Direct link to Negative Relative: Overlap")

Use a negative, relative value ("-=X") to place your tween X seconds before previous animations end

0

1

2blueSpin

3

4

Play

```
```

### Absolute: Anywhere[​](#absolute-anywhere "Direct link to Absolute: Anywhere")

Use an absolute value (number) to specify the exact time in seconds a tween should start.

0

1

2blueSpin

3

4

Play

```
```

### Labels[​](#labels "Direct link to Labels")

Use a label ("string") to specify where a tween should be placed.

0

1

2blueSpin

3

4

Play

```
```

### Relative to other tweens[​](#relative-to-other-tweens "Direct link to Relative to other tweens")

Use `"<"` to reference the most recently-added animation's START time. Use `">"` to reference the most recently-added animation's END time.

0

1

2blueSpin

3

4

Play

```
```

## Percentage-based values[​](#percentage-based-values "Direct link to Percentage-based values")

tip

As of GSAP 3.7.0, you can use percentage-based values, as explained in this video:

## Interactive Playground[​](#interactive-playground "Direct link to Interactive Playground")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/46cc11b9e4319e0562c580669b14c330?default-tab=result\&theme-id=41164)

Hopefully by now you can see the true power and flexibility of the `position` parameter. And again, even though these examples focused mostly on [timeline.to()](/docs/v3/GSAP/Timeline/to\(\).md), it works exactly the same way in [timeline.from()](http://docs/v3/GSAP/Timeline/from\(\)), [timeline.fromTo()](/docs/v3/GSAP/Timeline/fromTo\(\).md), [timeline.add()](/docs/v3/GSAP/Timeline/add\(\).md), [timeline.call()](/docs/v3/GSAP/Timeline/call\(\).md), and [timeline.addPause()](/docs/v3/GSAP/Timeline/addPause\(\).md).

\*Percentage-based values were added in GSAP 3.7.0<br /><!-- -->\*\*The "previous animation" refers to the most recently-inserted animation, not necessarily the animation that is closest to the end of the timeline.
