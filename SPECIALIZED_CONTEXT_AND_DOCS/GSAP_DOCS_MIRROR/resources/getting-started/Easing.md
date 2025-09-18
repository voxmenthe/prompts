# Easing

**Easing is possibly the most important part of motion design**. A well-chosen ease will add personality and breathe life into your animation.

Take a look at the difference between no ease and a bounce ease in the demo below! The green box with no ease spins around at a consistent speed, whereas the purple box with the 'bounce' ease revs up, races along and then bounces to a stop.

```
gsap.to(".green", { rotation: 360, duration: 2, ease: "none" });

gsap.to(".purple", { rotation: 360, duration: 2, ease: "bounce.out" });
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/gOvdzLK?default-tab=result\&theme-id=41164)

Under the hood, the "ease" is a mathematical calculation that controls the rate of change during a tween. But don't worry, we do all the math for you! You just sit back and select the ease that best fits your animation.

To use the ease visualizer, simply click on the ease name that you'd like to use. You can also click on the underlined text to change the values and type of ease.

## Ease Visualizer

value: 0.00

progress: 0.00

power0

power1

power2

power3

power4

Preview

graph

Show editing hints

* Add point: ALT-CLICK on line
* Toggle smooth/corner: ALT-CLICK anchor
* Get handle from corner anchor: ALT-DRAG
* Toggle select: SHIFT-CLICK anchor
* Delete anchor: press DELETE key
* Undo: CTRL-Z

#### [Core](https://gsap.com/core/)

none

power1

out

power2

power3

power4

back

bounce

circ

elastic

expo

sine

steps

#### Ease pack

rough

slow

expoScale

#### [Extra Eases](https://gsap.com/pricing/)

CustomEase

CustomBounce

CustomWiggle

Share Ease

// click to modify the underlined values

gsap.to(target,

<!-- -->

{

duration:2.5,

ease: "Cubic/power2 (power2).out",none",(

<!-- -->

{

template:none/linear (none).outnone,

strength: 1,

points:20,

taper:none,

randomize:\[x],

clamp:\[ ]

}

<!-- -->

)",

(0.7,0.7,\[ ])",(scale from 0.5 to 7 (0.5,7),none)",(12)",(1,0.3)",(1.7)",create("custom", ""0""),create("myWiggle",

<!-- -->

{

wiggles:10,

type:easeInOut

}

<!-- -->

),

create("myBounce",

<!-- -->

{

strength:0.7,

endAtStart:false,

squash:1,

squashID: "myBounce-squash"

}

<!-- -->

),

y: -500

rotation: 360

x: "400%"

}

<!-- -->

);

none (linear)

nonenonenone

power1

outinOutin

power2

outinOutin

power3

outinOutin

power4

outinOutin

back

outinOutin

elastic

outinOutin

bounce

outinOutin

Other

roughslowsteps

circ

outinOutin

expo

outinOutin

sine

outinOutin

Video Walkthrough

<br />

Huge thanks to Carl for providing this video. We highly recommend their extensive GSAP training at [CreativeCodingClub.com](https://www.creativecodingclub.com/bundles/creative-coding-club?ref=44f484). Enroll today in their [Free GSAP course](https://www.creativecodingclub.com/courses/FreeGSAP3Express?ref=44f484) and discover the joy of animating with code.

## Ease types[​](#ease-types "Direct link to Ease types")

For most eases you'll be able to specify a type. There are three types of ease: `in`, `out` and `inOut`. These control the momentum over the course of the ease.

tip

Ease **out** animations like `"power1.out"` are the best for UI transitions; they're fast to start which helps the UI feel responsive, and then they ease out towards the end giving a natural feeling of friction.

```
ease: "power1.in"
// start slow and end faster, like a heavy object falling

ease: "power1.out"
// start fast and end slower, like a rolling ball slowly coming to a stop

ease: "power1.inOut"
// start slow and end slow, like a car accelerating and decelerating
```

Can't get your ease working?

["SlowMo"](/docs/v3/Eases/SlowMo.md) ease, ["RoughEase"](/docs/v3/Eases/RoughEase.md), ["ExpoScaleEase"](/docs/v3/Eases/ExpoScaleEase.md), and custom eases (["CustomEase"](/docs/v3/Eases/CustomEase.md), ["CustomBounce"](/docs/v3/Eases/CustomBounce.md), and ["CustomWiggle"](/docs/v3/Eases/CustomWiggle.md)) are not in the core. **They must be loaded separately.** See the [installation page](/docs/v3/Installation) for details.
