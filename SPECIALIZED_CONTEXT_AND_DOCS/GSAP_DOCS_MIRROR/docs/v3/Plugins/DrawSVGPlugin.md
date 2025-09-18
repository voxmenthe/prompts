# DrawSVG

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(DrawSVGPlugin) 
```

#### Minimal usage

```
//draws all elements with the "draw-me" class applied 
gsap.from(".draw-me", {duration:1,drawSVG: 0});
```

### Description[​](#description "Direct link to Description")

DrawSVGPlugin allows you to progressively reveal (or hide) the stroke of an SVG `<path>`, `<line>`, `<polyline>`, `<polygon>`, `<rect>`, or `<ellipse>`. You can even animate outward from the center of the stroke (or any position/segment). It does this by controlling the `stroke-dashoffset` and `stroke-dasharray` CSS properties.

Think of the drawSVG value as describing the stroked portion of the overall SVG element (which doesn't necessarily have to start at the beginning). For example, `drawSVG:"20% 80%"` renders the stroke between the 20% and 80% positions, meaning there's a 20% gap on each end. If you started at `"50% 50%"` and animated to `"0% 100%"`, it would draw the stroke from the middle outward to fill the whole path.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/d99f307bef8b65451613ba899154515b?default-tab=result\&theme-id=41164)

Remember, the `drawSVG` value doesn't describe the values between which you want to animate - it describes the end state to which you're animating (or the beginning if you're using a `from()` tween). So `gsap.to("#path", {duration: 1, drawSVG: "20% 80%"})` animates it from wherever the stroke is currently to a state where the stroke exists between the 20% and 80% positions along the path. It does **NOT** animate it from 20% to 80% over the course of the tween.

This is a **good** thing because it gives you much more flexibility. You're not limited to starting out at a single point along the path and animating in one direction only. You control the whole segment (starting and ending positions). So you could even animate a dash from one end of the path to the other, never changing size, like `gsap.fromTo("#path", {drawSVG: "0 5%"}, {duration: 1, drawSVG: "95% 100%"});`

You may use either percentages or absolute lengths. If you use a single value, `0` is assumed for the starting value, so `"100%"` is the same as `"0 100%"` and `"true"`.

**IMPORTANT:** In order to animate the stroke, you must first actually apply one using either CSS or SVG attributes:

```
/* Define a stroke and stroke-width in CSS: */
.yourPath { 
  stroke-width: 10px;
  stroke: red;
}

/* or as SVG attributes: */
<path ... stroke-width="10px" stroke="red" />
```

Detailed walkthrough

[YouTube video player](https://www.youtube.com/embed/6UAoyBcn2fk)

### How do I animate many strokes and stagger animations?[​](#how-do-i-animate-many-strokes-and-stagger-animations "Direct link to How do I animate many strokes and stagger animations?")

The great thing about having DrawSVGPlugin integrated into GSAP is that you can tap into the rich API to quickly create complex effects and have total control (`pause`, `resume`, `reverse`, `seek`, nesting, etc.). So let's say you have 20 SVG elements that all have the class `draw-me` applied to them, and you want to draw them in a staggered fashion. You could do:

```
//draws all elements with the "draw-me" class applied with staggered start times 0.1 seconds apart
gsap.from(".draw-me", {duration:1,stagger: 0.1, drawSVG: 0});
```

Or you could create a timeline and drop the tweens into it so that you can control the entire sequence as a whole:

```
var tl = gsap.timeline();
tl.from(".draw-me", { duration: 2, drawSVG: 0 }, 0.1); 

//now we can control it:
tl.pause();
tl.play();
tl.reverse();
tl.seek(0.5);...
```

### Add `"live"` to recalculate length throughout animation[​](#add-live-to-recalculate-length-throughout-animation "Direct link to add-live-to-recalculate-length-throughout-animation")

In rare situations, the length of the SVG element itself may change during the drawSVG animation (like if the window is resized and things are responsive). In that case, you can simply append `"live"` to the value which will cause DrawSVGPlugin to update the length on every tick of the animation. So, for example, `drawSVG: "20% 70% live"`.

### Splitting a multi-segment `<path>`[​](#splitting-a-multi-segment-path "Direct link to splitting-a-multi-segment-path")

Usually it's best to use DrawSVGPlugin on `<path>` elements that are just one segment (doesn't contain multiple "M" commands) because browsers have a hard time properly rendering a single stroke through multiple segments, but we've crafted a helper function that automatically splits a multi-segment `<path>` into a `<path>` for each segment, as seen in this demo:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/yLEzJNE?default-tab=result\&theme-id=41164)

### Caveats / Notes[​](#caveats--notes "Direct link to Caveats / Notes")

* DrawSVGPlugin does not animate the fill of the SVG at all - it only affects the stroke using `stroke-dashoffset` and `stroke-dasharray` CSS properties.

* In some rare situations, Firefox doesn't properly calculate the total length of `<path>` elements, so you may notice that the path stops a bit short even if you animate to 100%. In this (uncommon) scenario, there are two solutions: either add more anchors to your path to make the control points hug closer to the path, or overshoot the percentage a bit, like use `102%` instead of `100%`. To be clear, this is a Firefox bug, not a bug with DrawSVGPlugin.

* As of December 2014, iOS Safari has a bug that causes it to render `<rect>` strokes incorrectly in some cases (too thick, and slight artifacts around the edges, plus it misplaces the origin). The best workaround is to either convert your `<rect>` to a `<path>` or `<polyline>`.

* You cannot affect the contents of a `<use>` element because browsers simply don't allow it. Well, you can tween them but you won't see any changes on the screen.

## **Methods**[​](#methods "Direct link to methods")

|                                                                                                                                            |                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #### [DrawSVGPlugin.getLength](/docs/v3/Plugins/DrawSVGPlugin/static.getLength\(\).md)( element:\[Element \| Selector text] ) : Number     | Provides an easy way to get the length of an SVG element's stroke including: `<path>`, `<rect>`, `<circle>`, `<ellipse>`, `<line>`, `<polyline>`, and `<polygon>` |
| #### [DrawSVGPlugin.getPosition](/docs/v3/Plugins/DrawSVGPlugin/static.getPosition\(\).md)( element:\[Element \| Selector text] ) : Number | Provides an easy way to get the current position of the DrawSVG.                                                                                                  |

## **Demos**[​](#demos "Direct link to demos")

Check out the full collection of [How-to demos](https://codepen.io/collection/XRqLgd) and our favourite [inspiring community demos](https://codepen.io/collection/DYmKKD) on CodePen.

DrawSVG Demos

Search..

\[x]All

Play Demo videos\[ ]

