# CustomEase

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(CustomEase) 
```

#### Minimal usage

```
CustomEase.create(
  "hop",
  "M0,0 C0,0 0.056,0.442 0.175,0.442 0.294,0.442 0.332,0 0.332,0 0.332,0 0.414,1 0.671,1 0.991,1 1,0 1,0"
);

//now you can reference the ease by ID (as a string):
gsap.to(element, { duration: 1, y: -100, ease: "hop" });
```

### Description[​](#description "Direct link to Description")

CustomEase frees you from the limitations of canned easing options; create literally any easing curve imaginable by simply drawing it in the [Ease Visualizer](/docs/v3/Eases.md) or by copying/pasting an SVG path. Zero limitations. Use as many control points as you want.

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

Creating a Custom Ease

How to use this ease visualizer

* **Add points** - ATL/OPTION-click anywhere on the curve
* **Delete points** - Select the point and then press the DELETE key on the keyboard
* **Toggle smooth/corner** - ALT/OPTION-click on an anchor point. Or, ALT/OPTION-drag a control handle to turn it into a corner (not smooth) point.
* **Select multiple points** - Hold the SHIFT key while clicking anchor points.
* **Undo** - Press CTRL-Z
* **Disable** snapping - Hold SHIFT while dragging

You can edit any of the other eases by selecting them and then hiting "CustomEase".

## Copy/Paste SVG[​](#copypaste-svg "Direct link to Copy/Paste SVG")

When in the "custom" mode of the Ease Visualizer, you can select the purple text at the bottom (the CustomEase data string), highlight it all, and then paste in an SVG path (like from Adobe Illustrator) and then click elsewhere and the Ease Visualizer will grab the first `<path>` and convert it into the proper format.

## Using cubic-bezier values[​](#using-cubic-bezier-values "Direct link to Using cubic-bezier values")

CustomEase also recognizes standard `cubic-bezier()` strings containing four numbers, like those you can get from [cubic-bezier.com](//cubic-bezier.com/). For example, `".17,.67,.83,.67"`. Either paste that into the orange text area in the bottom of the Ease Visualizer or feed it directly into the `CustomEase.create()` method, like `CustomEase.create("easeName", ".17,.67,.83,.67");`.

## The code[​](#the-code "Direct link to The code")

Instead of using the long data string in each tween, you simply `create()` a CustomEase once (typically as soon as your page/app loads) and assign it a memorable ID (like `"hop"` or `"wiggle"` or whatever you want) that you reference thereafter in any of your tweens, like:

```
//define your CustomEase and give it an ID ("hop" in this case)

CustomEase.create(
  "hop",
  "M0,0 C0,0 0.056,0.442 0.175,0.442 0.294,0.442 0.332,0 0.332,0 0.332,0 0.414,1 0.671,1 0.991,1 1,0 1,0"
);

//now you can reference the ease by ID (as a string):
gsap.to(element, { duration: 1, y: -100, ease: "hop" });
```

Creating the ease(s) initially ensures maximum performance during animation because there's some overhead involved in calculating all the points internally and optimizing the data for blisteringly fast runtime performance. That only happens once, upon creation.

Typically the path string uses normalized values (0-1), but you can pass in any SVG path data that uses cubic bezier instructions ("M", "C", "S", "L", or "Z" commands) and it'll normalize things internally.

## .getSVGData()[​](#getsvgdata "Direct link to .getSVGData()")

CustomEase has a `getSVGData()` method that calculates the SVG `<path>` data string for visualizing any ease graphically at any size that you define, like `{width: 500, height: 400, x: 10, y: 50}`. You can supply a CustomEase or the ID associated with one, or even a standard ease like `"power2"`. Feed in a `path` in the vars object and it'll populate its `d` attribute for you, like:

```
//create a CustomEase with an ID of "hop"
CustomEase.create(
  "hop",
  "M0,0 C0,0 0.056,0.445 0.175,0.445 0.294,0.445 0.332,0 0.332,0 0.332,0 0.414,1 0.671,1 0.991,1 1,0 1,0"
);

//draw the ease visually in the SVG  that has an ID of "ease" at 500px by 400px:
CustomEase.getSVGData("hop", { width: 500, height: 400, path: "#ease" });
```

## Naming caveat[​](#naming-caveat "Direct link to Naming caveat")

It's usually not a good idea to name your ease (the string name you associate with it) the same as one of the standard eases, like "expo" or "power1", etc. because that would essentially overwrite that standard ease and replace it with your CustomEase.

## **Demos**[​](#demos "Direct link to demos")

* [CustomEase demos](https://codepen.io/collection/AQKMdx)

## Videos[​](#videos "Direct link to Videos")

Overview

[YouTube video player](https://www.youtube.com/embed/A9ROywSFFiY)

Using CustomEase in a project

[YouTube video player](https://www.youtube.com/embed/rJRrUHds7fc)

## FAQ[​](#faq "Direct link to FAQ")

#### How do I include undefined in my project?

See the [installation page](/docs/v3/Installation) for all the options (CDN, NPM, download, etc.) where there's even an interactive helper that provides the necessary code. Easy peasy. Don't forget to [register undefined](/docs/v3/GSAP/gsap.registerPlugin\(\).md) like this in your project:

```
gsap.registerPlugin(undefined)
```

#### Is this included in the GSAP core?

No, you must load/import it separately

#### It works fine during development, but suddenly stops working in the production build! What do I do?

Your build tool is probably dropping the plugin when [tree shaking](https://developer.mozilla.org/en-US/docs/Glossary/Tree_shaking) and you forgot to [register undefined](/docs/v3/GSAP/gsap.registerPlugin\(\).md) (which protects it from tree shaking). Just register the plugin like this:

```
gsap.registerPlugin(undefined)
```

#### Is it bad to register a plugin multiple times?

No, it's perfectly fine. It doesn't help anything, nor does it hurt.
