# Handling conflicting tweens

tip

Have you ever been in a situation with GSAP where you needed a higher level of control over conflicting tweens? If you're just creating linear, self-playing animations, the default overwrite mode of false will work just fine for you. However, in cases where you are creating tweens dynamically based on user interaction or random events you may need finer control over how conflicts are resolved.

Overwriting refers to how GSAP handles conflicts between **multiple tweens** on the **same properties** of the **same targets** at the **same time**. The video below explains GSAP's overwrite modes and provides visual examples of how they work.

Video Walkthrough

## GSAPs 3 Overwrite Modes[​](#gsaps-3-overwrite-modes "Direct link to GSAPs 3 Overwrite Modes")

* **`false`** (default): No overwriting occurs and multiple tweens can try to animate the same properties of the same target at the same time. One way to think of it is that the tweens remain "fighting each other" until one ends.
* **`true`**: Any existing tweens that are animating the same target (regardless of which properties are being animated) will be killed immediately.
* **`"auto"`**: Only the *conflicting* parts of an existing tween will be killed. If tween1 animates the x and rotation properties of a target and then tween2 starts animating only the x property of the same targets and `overwrite: "auto"` is set on the second tween, then the rotation part of tween1 will remain but the x part of it will be killed.

## Setting Overwrite Modes[​](#setting-overwrite-modes "Direct link to Setting Overwrite Modes")

```
// Set overwrite on a tween
gsap.to(".line", { x: 200, overwrite: true });

// Set overwrite globally for all tweens
gsap.defaults({ overwrite: true });

// Set overwrite for all tweens in a timeline
const tl = gsap.timeline({ defaults: { overwrite: true } });
```

Below is the demo used in the video. [Open it in a new tab](https://codepen.io/snorkltv/pen/XWNRXqd) to experiment with the different overwrite modes

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/XWNRXqd?default-tab=result\&theme-id=41164)

Hopefully this article helps you better understand how much control GSAP gives you. Overwrite modes are one of those features that you may not need that often, but when you do, they can save you hours of trouble writing your own solution.

tip

For more tips like this and loads of deep-dive videos designed to help you quickly master GSAP, check out this course bundle from our friends at [CreativeCodingClub.com](https://www.creativecodingclub.com/bundles/creative-coding-club?ref=44f484).
