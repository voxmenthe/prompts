# Easing

Plugin Eases

["slow"](/docs/v3/Eases/SlowMo.md), ["rough"](/docs/v3/Eases/RoughEase.md), and ["expoScale"](/docs/v3/Eases/ExpoScaleEase.md) eases are not in the core - they are packaged together in an **EasePack** file in order to minimize file size. ["CustomEase"](/docs/v3/Eases/CustomEase.md), ["CustomBounce"](/docs/v3/Eases/CustomBounce.md), and ["CustomWiggle"](/docs/v3/Eases/CustomWiggle.md) are packaged independently as well (not in the core).

See the [installation page](/docs/v3/Installation) for details.

<br />

**Easing is the primary way to change the timing of your tweens.** Simply changing the ease can adjust the entire feel and personality of your animation. There are infinite eases that you can use in GSAP so we created the visualizer below to help you choose exactly the type of easing that you need.

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

Coding tip - Default Easing

GSAP uses a default ease of `"power1.out"`. You can overwrite this in any tween by setting the `ease` property of that tween to another (valid) ease value. You can set a different default ease for GSAP by using [gsap.defaults()](/docs/v3/GSAP/gsap.defaults\(\).md). You can also set defaults for particular [timelines](/docs/v3/GSAP/Timeline.md).

```
gsap.defaults({
  ease: "power2.in",
  duration: 1,
});

gsap.timeline({defaults: {ease: "power2.in"}})
```

## How to use the Ease Visualizer[​](#how-to-use-the-ease-visualizer "Direct link to How to use the Ease Visualizer")

To use the ease visualizer, simply click on the ease name that you'd like to use. You can also click on the underlined text to change the values and type of ease.<br /><!-- -->Use the navigation links in the menu to the left for more information about complex eases.

Video Walkthrough

<br />

Huge thanks to Carl for providing this video. We highly recommend their extensive GSAP training at [CreativeCodingClub.com](https://www.creativecodingclub.com/bundles/creative-coding-club?ref=44f484). Enroll today in their [Free GSAP course](https://www.creativecodingclub.com/courses/FreeGSAP3Express?ref=44f484) and discover the joy of animating with code.
