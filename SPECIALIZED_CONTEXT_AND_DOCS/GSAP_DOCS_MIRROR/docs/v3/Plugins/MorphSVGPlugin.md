# MorphSVG

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(MorphSVGPlugin) 
```

#### Minimal usage

```
gsap.to("#circle", { duration: 1, morphSVG: "#hippo" });
```

Detailed walkthrough

[YouTube video player](https://www.youtube.com/embed/Uxa9sdaeyKM)

## Description[​](#description "Direct link to Description")

MorphSVGPlugin morphs SVG paths by animating the data inside the `d` attribute. You can morph a circle into a hippo with a single line of code:

```
gsap.to("#circle", { duration: 1, morphSVG: "#hippo" });
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ZbLarZ?default-tab=result\&theme-id=41164)

How does it work?

In this example, MorphSVGPlugin finds the path with the ID of "circle" and the path with the ID of "hippo" and automatically figures out how to add enough points to the circle to get a super smooth morph. It will rip through all that ugly path data, convert everything to cubic beziers, and dynamically subdivide them when necessary, adding points so that the beginning and ending quantities match (but visually it looks the same). It's all done seamlessly under the hood.

## Features[​](#features "Direct link to Features")

Feature Highlights

* Morph `<path>` data **even if the number (and type) of points is completely different** between the start and end shapes! Most other SVG shape morphing tools require that the number of points matches.
* Morph a `<polyline>` or `<polygon>` to a different set of points.
* Draw the resulting shape to `<canvas>` (via setting `MorphSVGPlugin.defaultRender` a render function - see [the defaultRender docs](/docs/v3/Plugins/MorphSVGPlugin/static.defaultRender.md) for more information).

**read more...**

* There's a utility function, `MorphSVGPlugin.convertToPath()`, that can convert primitive shapes like `<circle>`, `<rect>`, `<ellipse>`, `<polygon>`, `<polyline>`, and `<line>` directly into the equivalent `<path>` that looks identical to the original and is swapped right into the DOM.
* Optionally define a `shapeIndex` that controls how the points get mapped. This affects what the inbetween state looks like during animation.
* Use either linear interpolation (the default) or a `rotational` type to get more natural looking morphs.
* Instead of passing in a raw path data as text, you can simply feed in selector text or an element and the plugin will grab the data it needs from there, making workflow easier.

When only specifying a shape, MorphSVGPlugin can take a wide range of values.

```
//selector string
gsap.to("#circle", {duration: 1, morphSVG: "#hippo"});

//an SVG element
var endShape = document.getElementById("hippo");
gsap.to("#circle", {duration: 1, morphSVG: endShape);

//points for  or  elements:
gsap.to("#polygon", {duration: 2, morphSVG: "240,220 240,70 70,70 70,220"});

//strings for  elements
gsap.to("#path", {duration: 2, morphSVG: "M10 315 L 110 215 A 30 50 0 0 1 162.55 162.45 L 172.55 152.45 A 30 50 -45 0 1 215.1 109.9 L 315 10"});
```

note

If the shape you pass in is a `<rect>`, `<circle>`, `<ellipse>` (or similar), MorphSVGPlugin will internally create path data from those shapes.

## Config Object[​](#config-object "Direct link to Config Object")

MorphSVG can be used as either a shorthand for the shape (described below) or as a configuration object with any of the following properties:

* ### Property

  ### Description

  #### shape[](#shape)

  String | Selector | Element - The shape to morph to.

* #### type[](#type)

  "rotational" | "linear"

  By default, all of the anchors and control points in the shape are interpolated linearly (`type: "linear"`) which is usually fine but you can set `type: "rotational"` to make MorphSVG use rotation and length data for interpolation instead which can produce more natural morphs in some cases. It also completely eliminates any kinks that may form in otherwise smooth anchors mid-tween. To tap into this alternative style of morphing, just set type: "rotational" in the object:

  ```
  gsap.to("#shape1", {
   duration: 2, 
   morphSVG:{
    shape: "#shape2",
    type: "rotational"
   }
  })
  ```

  The concept is best understood visually, so here are some videos and demos...

  View More details

  [YouTube video player](https://www.youtube.com/embed/C-qo_aEAPp8)

  ### Interactive comparison of linear and rotational morphs

  [MorphSVG type:"rotational" for more natural morphs](https://codepen.io/GreenSock/embed/vvjOGq?default-tab=result\&editable=true\&theme-id=41164)

* #### origin[](#origin)

  string

  Sets the origin of rotation. The default is `50% 50%`. The format is either a string of two percentage values, or a string or four values if there are different values for the start and end shapes.

  To set your own origin:

  ```
  gsap.to("#shape1", {
    duration: 2,
    morphSVG: {
      shape: "#shape2",
      type: "rotational",
      origin: "20% 60%", //or "20% 60%,35% 90%" if there are different values for the start and end shapes.
    },
  });
  ```

  sometimes the rotations around a point look odd, In cases like this, it's best to experiment and set your own custom origin to improve things even more. We created a findMorphOrigin() utility function to help with this...

  View More details

  findMorphOrigin allows you to simply feed in a start and end shape and then it'll superimpose an origin that you can drag around and see exactly how it affects the morph! In the demo below, go into the JS panel and un-comment the findMorphIndex() line and you'll see exactly how this works. Drag the origin around and watch how it affects things. [MorphSVG: fixing type:"rotational" weirdness](https://codepen.io/GreenSock/embed/VqRVgr?default-tab=result\&editable=true\&theme-id=41164)

* #### shapeIndex[](#shapeIndex)

  number

  The shapeIndex property allows you to adjust how the points in the start shape are mapped. The following code will map the third point in the square to the first point in the star.

  ```
  gsap.to("#square", {
    duration: 1,
    morphSVG: { shape: "#star", shapeIndex: 3 },
  });
  ```

  View More details

  In order to prevent points from drifting wildly during the animation MorphSVGPlugin needs to find a point in the start path that is in close proximity to the first point in the end path. Once that point is found it will map the next point in the start path to the second point in the end path (and so on and so on).

  Due to the complexity of vector art there will be times that you may want to change which point in the start path gets mapped to the first point in the end path. This is where shapeIndex comes in.

  **Notes**

  * shapeIndex only works on closed paths.
  * If you supply a negative shapeIndex the start path will be completely reversed (which can be quite useful


* #### map[](#map)

  size | position | complexity

  If the sub-segments inside your path aren't matching up the way you hoped between the start and end shapes, you can use `map` to tell MorphSVGPlugin which algorithm to prioritize:

  * `"size"` (the default) - Attempts to match segments based on their overall size. If multiple segments are close in size, it'll use positional data to match them. This mode typically gives the most intuitive morphs.
  * `"position"` - Matches mostly based on position.
  * `"complexity"` - Matches purely based on the quantity of anchor points. This is the fastest algorithm and it can be used to "trick" things to match up by manually adding anchors in your SVG authoring tool so that the pieces that you want matched up contain the same number of anchors (though that's completely optional).

  ```
  gsap.to("#id", {
    duration: 1,
    morphSVG: { shape: "#otherID", map: "complexity" },
  });
  ```

  View More details

  #### Notes

  * `map` is completely optional. Typically the default mode works great.
  * If none of the map modes get the segments to match up the way you want, it's probably best to just split your path into multiple paths and morph each one. That way you get total control.

* #### Precision[](#Precision)

  number

  By default, MorphSVGPlugin will round values to 2 decimal places in order to maximize performance and reduce string length but you can set `precision` to the number of decimal places if you prefer something different. For example, `precision: 5` would round to 5 decimal places:

  ```
  gsap.to("#id", { morphSVG: { shape: "#other-id", precision: 5 } });
  ```

* #### render[](#render)

  function

  Define a render function that'll be called every time the path updates. For more information see [Rendering to canvas](#rendering-to-canvas)

* #### precompile[](#precompile)

  Array

  Tell MorphSVGPlugin to run all of its initial calculations and return an array with the transformed strings, logging them to the console where you can copy and paste them back into your tween. That way, when the tween begins it can just grab all the values directly instead of doing expensive calculations.

  For more information see [precompile](#precompile)

## Tips[​](#tips "Direct link to Tips")

note

MorphSVG also stores the original path data on any target of a morph tween so that you can easily tween back to the original shape. (like `data-original="M490.1,280.649c0,44.459-36.041,80..."`

### findShapeIndex() util[​](#findshapeindex-util "Direct link to findShapeIndex() util")

Experimenting with `shapeIndex` can be a bit of a guessing game. To make things easier we have created a stand-alone utility function called `findShapeIndex()`. This function provides an interactive user interface to help you visualize where the start point is, change it and preview the animation.

You can load `findShapeIndex()` from [this download link](https://s3-us-west-2.amazonaws.com/s.cdpn.io/16327/findShapeIndex.js).

Once it's loaded you simply tell it which shapes to use.

```
findShapeIndex("#square", "#star");
```

Or pass in raw data:

```
findShapeIndex(
  "#square",
  "M10 315 L 110 215 A 30 50 0 0 1 162.55 162.45 L 172.55 152.45 A 30 50 -45 0 1 215.1 109.9 L 315 10"
);
```

tip

The best way to get started is to drop your SVG into the pen and alter the IDs to match your svg.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/763b6533f17a795c3cd957c668c33882?default-tab=result\&theme-id=41164)

### Converting SVG shapes to paths[​](#converting-svg-shapes-to-paths "Direct link to Converting SVG shapes to paths")

Feature runthrough

[YouTube video player](https://www.youtube.com/embed/jcq9kEyJNMM)

Technically it's only feasible to morph `<path>` elements or `<polyline>`/`<polygon>` elements, but what if you want to morph a `<circle>`, `<rect>`, `<ellipse>`, or `<line>`? No problem - just tap into the utility method and have the plugin do the conversion for you:

```
MorphSVGPlugin.convertToPath("#elementID");
```

You can pass in an element or selector text, so you could also have it convert ALL of those elements with one line:

```
MorphSVGPlugin.convertToPath("circle, rect, ellipse, line, polygon, polyline");
```

This literally swaps in a for each one directly in the DOM, and it should look absolutely identical. It'll keep the attributes, like the "id" attribute. So after the conversion, you should be able to target the elements pretty easily, just as you would before.

```
//An svg <rect> Like this:
<rect id="endShape" width="100" height="100" fill="red"/>
//becomes
<path id="endShape" fill="red" d="M100,0 v100 h-100 v-100 h100z"></path>
```

### Morph into multiple shapes[​](#morph-into-multiple-shapes "Direct link to Morph into multiple shapes")

Since MorphSVGPlugin is so tightly integrated into GSAP, sequencing multiple morphs is a breeze. Watch how easy it is to make that circle morph into a hippo, star, elephant and back to a circle.

```
tl.to(circle, { duration: 1, morphSVG: "#hippo" }, "+=1")
  .to(circle, { duration: 1, morphSVG: "#star" }, "+=1")
  .to(circle, { duration: 1, morphSVG: "#elephant" }, "+=1")
  .to(circle, { duration: 1, morphSVG: circle }, "+=1");
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/rOjeRq?default-tab=result\&theme-id=41164)

## Performance[​](#performance "Direct link to Performance")

### Define a shapeIndex in advance[​](#define-a-shapeindex-in-advance "Direct link to Define a shapeIndex in advance")

Performance tip: define a `shapeIndex` in advance

MorphSVGPlugin's default `shapeIndex: "auto"` does a bunch of calculations to reorganize the points so that they match up in a natural way but if you define a numeric `shapeIndex` (like `shapeIndex: 5`) it skips those calculations. Each segment inside a path needs a `shapeIndex`, so multiple values are passed in an array like `shapeIndex:[5, 1, -8, 2]`. But how would you know what numbers to pass in? The `findShapeIndex()` tool helps for single-segment paths, what about multi-segment paths? It's a pretty complex thing to provide a GUI for.

**read more...**

Typically the default `"auto"` mode works great but the goal here is to avoid the calculations, so there is a `"log"` value that will act just like `"auto"` but it will also `console.log()` the `shapeIndex` value(s). That way, you can run the tween in the browser once and look in your console and see the numbers that `"auto"` mode would produce. Then it's simply a matter of copying and pasting that value into your tween where `"log"` was previously.

For example:

```
//logs a value like "shapeIndex:[3]"
gsap.to("#id", {
  duration: 1,
  morphSVG: { shape: "#otherID", shapeIndex: "log" },
});
//now you can grab the value from the console and drop it in...
gsap.to("#id", {
  duration: 1,
  morphSVG: { shape: "#otherID", shapeIndex: [3] },
});
```

### Precompile[​](#precompile "Direct link to Precompile")

Performance tip: precompile

The biggest performance improvement comes from precompiling which involves having MorphSVGPlugin run all of its initial calculations listed above and then spit out an array with the transformed strings, logging them to the console where you can copy and paste them back into your tween. That way, when the tween begins it can just grab all the values directly instead of doing expensive calculations.

**show example...**

```
//logs a value like precompile:["M0,0 C100,200 120,500 300,145 34,245 560,46","M0,0 C200,300 100,400 230,400 100,456 400,300"]
gsap.to("#id", {
  duration: 1,
  morphSVG: { shape: "#otherID", precompile: "log" },
});
//now you can grab the value from the console and drop it in...
gsap.to("#id", {
  duration: 1,
  morphSVG: {
    shape: "#otherID",
    precompile: [
      "M0,0 C100,200 120,500 300,145 34,245 560,46",
      "M0,0 C200,300 100,400 230,400 100,456 400,300",
    ],
  },
});
```

As an example, here's [a really cool CodePen](https://codepen.io/davatron5000/pen/meNOqK/) by Dave Rupert before it was precompiled. Notice the very first time you click the toggle button, it may seem to jerk a bit because the entire brain is one path with many segments, and it must get matched up with all the letters and figure out the `shapeIndex` for each (which is expensive). By contrast, [here's a fork](https://codepen.io/GreenSock/pen/MKevzM) of that pen that has precompile enabled. You may noticed that it starts more smoothly.

#### Notes[​](#notes "Direct link to Notes")

* `precompile` is only available on `<path>` elements (not `<polyline>`/`<polygon>`). You can easily convert things using `MorphSVGPlugin.convertToPath("polygon, polyline");`

* Precompiling only improves the performance of the first (most expensive) render. If your entire morph is janky throughout the tween, it most likely has nothing to do with GSAP; your SVG may be too complex for the browser to render fast enough. In other words, the bottleneck is probably the browser's graphics rendering routines. Unfortunately, there's nothing GSAP can do about that and you'll need to simplify your SVG artwork and/or reduce the size at which it is displayed.

* The precompiled values are inclusive of `shapeIndex` adjustments. In other words, `shapeIndex` gets baked in.

* In most cases, you probably don't need to precompile; it's intended to be an advanced technique for squeezing every ounce of performance out of a very complex morph.

* If you alter the original start or end shape/artwork, make sure you precompile again so that the values reflect your changes.

### Rendering to canvas[​](#rendering-to-canvas "Direct link to Rendering to canvas")

Performance tip: Canvas

SVG is fantastic, but sometimes developers have a canvas-based project (often for rendering performance reasons). The MorphSVG plugin allows you to define a `render` function that'll be called every time the path updates, and it will receive two parameters:

**read more...**

1. **`rawPath`** \[array]: A RawPath is essentially an array containing an array for each contiguous segment with alternating x, y, x, y cubic bezier data. It's like an SVG `<path>` where there's one segment (array) for each `M` command. That segment (array) contains all of the cubic bezier coordinates in alternating x/y format (just like SVG path data) in raw numeric form which is nice because that way you don't have to parse a long string and convert things.

   For example, this SVG `<path>` has two separate segments because there are two `M` commands: `<path d="M0,0 C10,20,15,30,5,18 M0,100 C50,120,80,110,100,100"></path>` So the resulting **RawPath** would be:

   ```
   [
     [0, 0, 10, 20, 15, 30, 5, 18],
     [0, 100, 50, 120, 80, 110, 100, 100],
   ];
   ```

   For simplicity, the example above only has one cubic bezier in each segment, but there could be an unlimited quantity inside each segment. No matter what path commands are in the original`<path>` data string (cubic, quadratic, arc, lines, whatever), the resulting RawPath will **ALWAYS** be cubic beziers.

2. **`target`** \[object]: The target of the tween (usually a `<path>`)

This means you can even render morphs to super high-performance engines like [PixiJS](//pixijs.com/) or anything that'll allow you to draw cubic beziers!

### Demo: MorphSVG canvas rendering[​](#demo-morphsvg-canvas-rendering "Direct link to Demo: MorphSVG canvas rendering")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/WYWZab?default-tab=result\&theme-id=41164)

Here's an example of a tween and a render function that'd draw the morphing shape to canvas:

```
let canvas = document.querySelector("canvas"),
  ctx = canvas.getContext("2d"),
  vw = (canvas.width = window.innerWidth),
  vh = (canvas.height = window.innerHeight);
ctx.fillStyle = "#ccc";
gsap.to("#hippo", {
  duration: 2,
  morphSVG: {
    shape: "#circle",
    render: draw,
  },
});
function draw(rawPath, target) {
  let l, segment, j, i;
  ctx.clearRect(0, 0, vw, vh);
  ctx.beginPath();
  for (j = 0; j < rawPath.length; j++) {
    segment = rawPath[j];
    l = segment.length;
    ctx.moveTo(segment[0], segment[1]);
    for (i = 2; i < l; i += 6) {
      ctx.bezierCurveTo(
        segment[i],
        segment[i + 1],
        segment[i + 2],
        segment[i + 3],
        segment[i + 4],
        segment[i + 5]
      );
    }
    if (segment.closed) {
      ctx.closePath();
    }
  }
  ctx.fill("evenodd");
}
```

To set a default render method for all tweens:

```
MorphSVGPlugin.defaultRender = yourFunction;
```

### `updateTarget: false`[​](#updatetarget-false "Direct link to updatetarget-false")

By default, MorphSVG will update the original target of the tween (typically an SVG `<path>` element), but if you're only drawing to canvas you can tell MorphSVG to skip updating the target like this:

```
gsap.to("#hippo", {
  duration: 2,
  morphSVG: {
    shape: "#circle",
    render: draw,
    updateTarget: false,
  },
});
```

To set the default `updateTarget` value for all tweens (so that you don't have to add it to every tween):

```
MorphSVGPlugin.defaultUpdateTarget = false;
```

## Video Walkthroughs[​](#video-walkthroughs "Direct link to Video Walkthroughs")

### Advanced features for tricky morphs[​](#advanced-features-for-tricky-morphs "Direct link to Advanced features for tricky morphs")

Feature walkthrough

[YouTube video player](https://www.youtube.com/embed/CbzqU3xBvjg)

### Performance tips[​](#performance-tips "Direct link to Performance tips")

Performance tips

[YouTube video player](https://www.youtube.com/embed/n_5tx2onBzE)

## **Properties**[​](#properties "Direct link to properties")

|                                                                                                                    |                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #### [MorphSVGPlugin.defaultRender](/docs/v3/Plugins/MorphSVGPlugin/static.defaultRender.md) : Function            | Sets the default function that should be called whenever a morphSVG tween updates. This is useful if you're rendering to `<canvas>`.                                              |
| #### [MorphSVGPlugin.defaultType](/docs/v3/Plugins/MorphSVGPlugin/static.defaultType.md) : String                  | Sets the default `"type"` for all MorphSVG animations. The default `type` is `"linear"` but you can change it to `"rotational"`.                                                  |
| #### [MorphSVGPlugin.defaultUpdateTarget](/docs/v3/Plugins/MorphSVGPlugin/static.defaultUpdateTarget.md) : Boolean | Sets the default `updateTarget` value for all MorphSVG animations; if `true`, the original tween target (typically an SVG `<path>` element) itself gets updated during the tween. |

## **Methods**[​](#methods "Direct link to methods")

|                                                                                                                                                  |                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #### [MorphSVGPlugin.convertToPath](/docs/v3/Plugins/MorphSVGPlugin/static.convertToPath.md)( shape:\[Element \| String], swap:Boolean ) : Array | Converts SVG shapes like `<circle>`, `<rect>`, `<ellipse>`, or `<line>` into `<path>`                                                                                                                                                                                                                       |
| #### [MorphSVGPlugin.rawPathToString](/docs/v3/Plugins/MorphSVGPlugin/static.rawPathToString.md)( rawPath:Array ) : String                       | Converts a RawPath (array) into a string of path data, like `"M0,0 C100,20 300,50 400,0..."` which is what's typically found in the `d` attribute of a `<path>`.                                                                                                                                            |
| #### [MorphSVGPlugin.stringToRawPath](/docs/v3/Plugins/MorphSVGPlugin/static.stringToRawPath.md)( data:String ) : RawPath                        | Takes a string of path data (like `"M0,0 C100,20 300,50 400,0..."`, what's typically found in the `d` attribute of a `<path>`), parses it, converts it into cubic beziers, and returns it as a RawPath which is just an array containing an array for each segment (each `M` command starts a new segment). |

## **Demos**[​](#demos "Direct link to demos")

Check out the full collection of [SVG animation demos](https://codepen.io/collection/NqewVd) on CodePen.

MorphSVG Demos

Search..

\[x]All

Play Demo videos\[ ]

