May 4, 2015

# SVG

info

When it comes to animation, SVG and GSAP go together like peanut butter and jelly. Chocolate and strawberries. Bacon and...anything. SVG offers the sweet taste of tiny file size plus excellent browser support and the ability to scale graphics infinitely without degradation. They're perfect for building a rich, responsive UI (which includes animation, of course).

However, just because every major browser offers excellent support for **displaying** SVG graphics doesn't mean that **animating** them is easy or consistent. Each browser has its own quirks and implementations of the SVG spec, causing quite a few challenges for animators. For example, some browsers don't support CSS animations on SVG elements. Some don't recognize CSS transforms (rotation, scale, etc.), and implementation of transform-origin is a mess.

Don't worry, GSAP smooths out the rough spots and harmonizes behavior across browsers for you. There are quite a few unique features that GSAP offers specifically for SVG animators. Below we cover some of the things that GSAP does for you and then we have a list of other things to watch out for. This page is intended to be a go-to resource for anyone animating SVG with GSAP.

## Challenges that GSAP solves for you[​](#challenges-that-gsap-solves-for-you "Direct link to Challenges that GSAP solves for you")

GSAP does the best that it can to normalize browser issues and provide useful tools to make animate SVG as easy as it can be. Here are some of the challenges that using GSAP to animate SVG solves for you:

### Scale, rotate, skew, and move using 2D transforms[​](#scale-rotate-skew-and-move-using-2d-transforms "Direct link to Scale, rotate, skew, and move using 2D transforms")

When using GSAP, 2D transforms on SVG content work exactly like they do on any other DOM element.

```
gsap.to("#gear", {duration: 1, x: 100, y: 100, scale: 0.5, rotation: 180, skewX: 45});
```

Since IE and Opera don't honor CSS transforms at all, GSAP applies these values via the SVG `transform` attribute like:

```
<g id="gear" transform="matrix(0.5, 0, 0, 0.5, 100, 0)">...</g>
```

When it comes to animating or even setting 2D transforms in IE, CSS simply is not an option.

```
#gear {
  /* won't work in IE */
  transform: translateX(100px) scale(0.5);
}
```

Very few JavaScript libraries take this into account, but GSAP handles this for you behind the scenes so you can get amazing results in IE with no extra hassles.

### Set the transformOrigin[​](#set-the-transformorigin "Direct link to Set the transformOrigin")

Another unique GSAP feature: use the same syntax you would with normal DOM elements and get the same behavior. For example, to rotate an SVG `<rect>` that is 100px tall by 100px wide around its center you can do *any* of the following:

```
gsap.to("rect", {duration: 1, rotation: 360, transformOrigin: "50% 50%"}); //percents
gsap.to("rect", {duration: 1, rotation: 360, transformOrigin: "center center"}); //keywords
gsap.to("rect", {duration: 1, rotation: 360, transformOrigin: "50px 50px"}); //pixels
```

The demo below shows complete parity between DOM and SVG when setting `transformOrigin` to various values. We encourage you to test it in all major browsers and devices.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/17b4f47aa28f02e305dab386e9f79e89?default-tab=result\&theme-id=41164)

According to the SVG spec, the transform origin of an element is relative to its parent SVG canvas. So if you want to rotate an SVG child element around its own center, you need to manually plot that point in relation to the top-left corner of the SVG canvas. This can be rather cumbersome with multiple elements or if you ever change the position of the element. In contrast, a DOM element's transform-origin is relative to **its own** top left corner. Most developers expect (and most likely appreciate) the behavior they are accustomed to in the DOM – transform-origin is relative to the element itself.

Among other transform-origin related browser bugs (like zooming in Safari) we also found that Firefox doesn't honor percentages or keyword-based values. To learn the technical details of how GSAP fixes these `transformOrigin` issues behind the scenes, check out [this CSS-Tricks article](https://css-tricks.com/svg-animation-on-css-transforms/).

### Set transformOrigin without unsightly jumps[​](#set-transformorigin-without-unsightly-jumps "Direct link to Set transformOrigin without unsightly jumps")

You can run into some unexpected (yet "according to spec") results when changing the `transformOrigin` *after* an element has been transformed. In simple terms, once you scale or rotate an SVG element and then change its `transformOrigin`, the new `transformOrigin` is aligned to where it would have been according to the elements *un-transformed* state. This forces the element to be re-positioned and then the transforms are applied – leading to some awkward results.

With `smoothOrigin` enabled CSSPlugin applies some offsets so that the transformOrigin will be placed where you want without any jumps. It's a tough concept to explain with mere words so we made a nice little video for you and an interactive demo.

Study the demo below and scrub slowly.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/053d0ee8da31db3bdca1a4531e0628ee?default-tab=result\&theme-id=41164)

When changing the `transformOrigin` (or `svgOrigin`) of an SVG element, CSSPlugin will automatically record/apply some offsets to ensure that the element doesn't "jump". You can disable this by setting `CSSPlugin.defaultSmoothOrigin = false`, or you can control it on a per-tween basis using `smoothOrigin:true | false`. `smoothOrigin` only applies to SVG elements.

### Transform SVG elements around any point in the SVG canvas[​](#transform-svg-elements-around-any-point-in-the-svg-canvas "Direct link to Transform SVG elements around any point in the SVG canvas")

Sometimes it's useful to define the transformOrigin in the SVG's global coordinate space rather than relative to the element itself. GSAP has you covered. Our CSSPlugin recognizes a `svgOrigin` special property that works exactly like `transformOrigin` but it uses the SVG's **global coordinate space** instead of the element's local coordinate space. This can be very useful if, for example, you want to make a bunch of SVG elements rotate around a common point.

```
//rotate svgElement as though its origin is at x:250, y:100 in the SVG canvas's global coordinates.
gsap.to(svgElement, {duration: 1, rotation: 270, svgOrigin: "250 100"}); 
```

The demo below shows how `transformOrigin` and `svgOrigin` compare.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/6c9c78077e09ff7a9dfdc59af29079cf?default-tab=result\&theme-id=41164)

For the majority of use cases where setting a point of origin is necessary, `transformOrigin` is ideal, as it delivers behavior consistent with normal DOM elements in all major browsers. No headaches. However, there will be times when `svgOrigin` will come in handy too. [Sara Soueidan](https://twitter.com/SaraSoueidan) used this feature in her excellent [Circulus tool demo](https://sarasoueidan.com/tools/circulus/). `svgOrigin` only supports px-based values (no percentages).

### Animate SVG attributes like cx, cy, radius, width, etc.[​](#animate-svg-attributes-like-cx-cy-radius-width-etc "Direct link to Animate SVG attributes like cx, cy, radius, width, etc.")

GSAP handles pretty much any CSS properties like `fill`, `stroke`, `strokeWeight`, `fillOpacity`, etc. but to animate attributes you can use the built in [AttrPlugin](/docs/v3/GSAP/CorePlugins/Attributes.md) which handles any numeric attribute. For example, let's say your SVG element looks like this:

```
<rect id="rect" fill="none" x="0" y="0" width="500" height="400"></rect>
```

You could tween the "x", "y", "width", or "height" attributes using AttrPlugin like this:

```
gsap.to("#rect", {duration: 1, attr: {x: 100, y: 50, width: 100, height: 100}, ease: "none"});
```

Check out the JS tab in the demo below to see the syntax used.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/907579916ec09b51ff63459d6894da5e?default-tab=result\&theme-id=41164)

If you want to specify a particular unit (px or percentages) in your AttrPlugin tween it is important that the element already use the same units. AttrPlugin doesn't do unit conversion (like between % and px).

```
//Given the following element with percentage-based position:
<rect id="rectangle" x="0%" y="0%" class="element" width="200" height="100"/>

//The following tween would put the top left corner of the  exactly in the center of its parent's coordinate space.
gsap.to("#rectangle", {duration: 1, attr: {x: "50%", y: "50%"});
```

By combining percentage-based positional attributes and percentage-based transforms you could center the `<rect>` in its parent like so:

```
gsap.to("#rectangle", {duration: 1, attr: {x: "50%", y: "50%"}, x: "-50%", y: "-50%"})
```

See the[demo](https://codepen.io/GreenSock/pen/2e6fc2f37f89a718cd79ac7981f67f3f?editors=001).

### Use percentage-based x/y transforms[​](#use-percentage-based-xy-transforms "Direct link to Use percentage-based x/y transforms")

Another "gotcha" in the world of SVG is that percentage-based transforms are not accounted for in the SVG spec. When building responsive sites it can be very handy to move or simply position an element based on a percentage of its own native width or height. In the demo below four boxes of varying widths are all translated along the x-axis based on 100% of their width. No need to manually plug in unique pixel values for unique tweens of each element. All the animation runs off 1 line of code (see the JS tab).

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/f3acfdfb2847f7f79d1d4817cc0f3ecd?default-tab=result\&theme-id=41164)

To make this happen GSAP converts the percentage values you provide to pixel values. There is one important caveat: the value are not “live” like normal percentage-based CSS transforms are. GSAP has to do the math to bake the pixel values into the matrix(), thus if you change the element's width or height AFTER you apply a GSAP percentage-based transform, the translation won't be adjusted.

### Drag SVG elements (with accurate bounds and hit-testing)[​](#drag-svg-elements-with-accurate-bounds-and-hit-testing "Direct link to Drag SVG elements (with accurate bounds and hit-testing)")

There are quite a few tools out there that allow you to drag DOM elements, but few are optimized for SVG elements. With GreenSock's [Draggable](/docs/v3/Plugins/Draggable/.md), you can drag and drop SVG elements, confine movement to any axis and/or within bounds, do hitTest() detection, and throw or spin them too (with [InertiaPlugin](/docs/v3/Plugins/InertiaPlugin/.md)). Impressive fact: It even works inside nested transformed elements. Each interactive element below is a `<g>` contained in a single SVG canvas.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/4e36243272077577366e8f5facbcc586?default-tab=result\&theme-id=41164)

## Move *anything* along a path[​](#move-anything-along-a-path "Direct link to move-anything-along-a-path")

Move *anything* (DOM, SVG) along a path including autorotation, offset, looping, and more

GSAP's [MotionPathPlugin](/docs/v3/Plugins/MotionPathPlugin.md) makes it a *breeze* to animate DOM or SVG elements along a path. It has a lot of features like autorotation, offset, and looping in additional to *very* useful helper functions for converting between coordinate systems.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/LwzMKL?default-tab=result\&theme-id=41164)

## Animate SVG strokes[​](#animate-svg-strokes "Direct link to Animate SVG strokes")

[DrawSVGPlugin](/docs/v3/Plugins/DrawSVGPlugin.md) allows you to progressively reveal (or hide) the **stroke** of an SVG `<path>`, `<line>`, `<polyline>`, `<polygon>`, `<rect>`, or `<ellipse>`. It does this by controlling the `stroke-dashoffset` and `stroke-dasharray` CSS properties. You can even animate the stroke in both directions.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/jEEoyw?default-tab=result\&theme-id=41164)

Unlike most "pure CSS" solutions that only allow you to animate what percentage of a stroke is visible, DrawSVGPlugin allows you to animate both the starting and ending positions of the stroke segment that is being animated. This allows you to make more advanced animations:

* Increase/reveal stroke segment from beginning, end, center or any position.
* Decrease/hide stroke segment to beginning, end, center or any position.
* Move a segment of a stroke along a path for that [snake in a maze effect](https://codepen.io/GreenSock/full/Qbjxmx/).

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/GJpGdm?default-tab=result\&theme-id=41164)

### Morph SVG paths with differing numbers of points[​](#morph-svg-paths-with-differing-numbers-of-points "Direct link to Morph SVG paths with differing numbers of points")

MorphSVGPlugin provides advanced control over tweens that morph SVG paths.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/rOjeRq?default-tab=result\&theme-id=41164)

With [MorphSVGPlugin](https://gsap.com/morphSVG), you can

* Morph `<path>` data **even if the number (and type) of points is completely different** between the start and end shapes! Most other SVG shape morphing tools require that the number of points matches.
* Morph a `<polyline>` or `<polygon>` to a different set of points
* Convert and replace non-path SVG elements (like `<circle>`, `<rect>`, `<ellipse>`, `<polygon>`, `<polyline>`, and `<line>`) into identical `<path>`s using MorphSVGPlugin.convertToPath().
* Optionally define a "shapeIndex" that controls how the points get mapped. This affects what the in-between state looks like during animation.
* Simply feed in selector text or an element (instead of passing in raw path data) and the plugin will grab the data it needs from there, making workflow easier.

## Tips to Avoid Common Gotchas[​](#tips-to-avoid-common-gotchas "Direct link to Tips to Avoid Common Gotchas")

There are some things that GSAP can't solve for you. But hopefully this part of the article can help prepare you to avoid them ahead of time! Here are some things to keep in mind when creating and animating SVGs.

Vector editor/SVG creation tips:

* When creating an SVG in Illustrator or Inkscape, create a rectangle the same size as your artboard for when you copy elements out of your vector editor and paste them into a code editor ([how-to here](https://www.motiontricks.com/use-background-rectangle-with-svg-exports/)).

* How to quickly reverse the direction of a path in Illustrator (Note: If the Path Direction buttons are not visible in the attributes panel, click the upper right menu of that panel and choose 'Show All'):

  <!-- -->

  * Open path: Select the pen tool and click on the first point of your path and it will reverse the points.
  * Closed path: Right click the path and make it a compound path, choose menu-window-attributes and then use the Reverse Path Direction buttons.

* If you're morphing between elements it might be useful to [add extra points](https://gsap.com/community/topic/13681-svg-gotchas/?do=findComment\&comment=119550) yourself to simpler shapes where necessary so that MorphSVG doesn't have to guess at where to add points.

* You can think of masks as clip-paths that allow for alpha as well.

* When using masks, it's often important to [specify which units to use](https://gsap.com/community/topic/13681-svg-gotchas/?do=findComment\&comment=65150).

* Use a tool like [SVGOMG](https://jakearchibald.github.io/svgomg/) (or [this simpler tool](https://petercollingridge.appspot.com/svg-editor)) to minify your SVGs before using them in your projects.

Code/animation-related tips:

* Always set transforms of elements with GSAP (not just CSS). There are quite a few browser bugs related to getting transform values of elements which GSAP can't fix or work around so you should always set the transform of elements with GSAP if you're going to animate that element with GSAP.
* Always use relative values when animating an SVG element. Using something like `y: "+=100"` allows you to change the SVG points while keeping the same animation effect as hard coding those values.
* You can fix some rendering issues (especially in Chrome) by adding a very slight rotation to your tween(s) like `rotation: 0.01`.
* If you're having performance issues with your issue, usually the issue is that you have too many elements or are using filters/masks too much. For more information, see [this post](https://codepen.io/tigt/post/improving-svg-rendering-performance) focused on performance with SVGs.
* You might like injecting SVGs into your HTML instead of keeping it there directly. You can do this by using a tool [like Gulp](https://github.com/kiyopikko/inject-inline-svg/blob/master/gulp/injectsvg.js).
* You can easily convert between coordinate systems by using MotionPathPlugin's helper functions like [.convertCoordinates()](/docs/v3/Plugins/MotionPathPlugin/static.convertCoordinates\(\).md).

Technique tips/resources:

* You can [animate the viewBox attribute](https://www.motiontricks.com/basic-svg-viewbox-animation/) ([demo](https://cdpn.io/PointC/pen/OMabPa))!

* You can animate (draw) a dashed line by following the technique outlined in [this post](https://www.motiontricks.com/svg-dashed-line-animation/).

* You can animate (draw) lines with varied widths by following the technique outlined in [this post](https://www.motiontricks.com/svg-calligraphy-handwriting-animation/).

* You can animate (draw) handwriting effects by following the technique outlined in [this post](https://www.motiontricks.com/animated-handwriting-effect-part-1/).

* You can [create dynamic SVG elements](https://www.motiontricks.com/creating-dynamic-svg-elements-with-javascript/)!

* You can animate (draw) [a "3D" SVG path](https://gsap.com/community/topic/13681-svg-gotchas/?tab=comments#comment-58229).

* You can fake nested SVG elements (which will be available in SVG 2) by positioning the inner SVG with GSAP and scaling it ([demo](https://cdpn.io/PointC/pen/BwwEyK)).

* You can fake 3D transforms (which will be available in SVG 2) in some cases by either

  <!-- -->

  * Faking the transform that you need. For example sometimes rotationYs can be replaced by a `scaleX` instead.
  * Applying the transform to a container instead. If you can limit the elements within the SVG to just the ones you want to transform, this is a great approach. For example, applying a `rotationY` to the `<svg>` or `<div>` containing a `<path>` instead of applying it to the `<path>` itself.

* [When to use SVG vs When to use canvas](https://css-tricks.com/when-to-use-svg-vs-when-to-use-canvas/), Performance deep dive on CSS tricks

## Limitations of SVG[​](#limitations-of-svg "Direct link to Limitations of SVG")

* The current SVG spec does not account for 3D transforms. Browser support is varied. Best to test thoroughly and have fallbacks in place.
* SVG is now [hardware accelerated](https://developer.chrome.com/blog/hardware-accelerated-animations/#hardware-accelerated-svg-animations) in most browsers, but certain properties (such as filters) are still very performance intensive to animate.

## Browser support[​](#browser-support "Direct link to Browser support")

All SVG features in this article will work in all major desktop and mobile browsers unless otherwise noted. If you find any cross-browser inconsistencies please don't hesitate to let us know in our [support forums](https://gsap.com/community/forum/11-gsap/).

## Inspiration[​](#inspiration "Direct link to Inspiration")

[![blog-svg-tips-full](/assets/images/blog-svg-tips-full-9a47f74f87cb6a078104514673327ddd.png)](https://codepen.io/collection/XzxeNJ/)<br /><!-- -->The [Chris Gannon GSAP Animation collection](https://codepen.io/collection/XzxeNJ/) is great for seeing more SVG animations made with GSAP. Be sure to also check out [Chris Gannon's full portfolio on CodePen](https://codepen.io/chrisgannon/) and [follow him on Twitter](https://twitter.com/ChrisGannon) for a steady influx of inspiration.

## Awesome SVG Resources[​](#awesome-svg-resources "Direct link to Awesome SVG Resources")

* [SVG Tutorials](https://www.motiontricks.com/svg-tutorials/) - MotionTricks
* [The SVG Animation Masterclass](https://www.cassie.codes/speaking/getting-started-with-svg-animation/) - Cassie Evans
* [Understanding SVG Coordinate Systems and Transformations](https://sarasoueidan.com/blog/svg-coordinate-systems/) - Sara Soueidan
* [Improving SVG Runtime Performance](https://codepen.io/tigt/post/improving-svg-rendering-performance) - Taylor Hunt
* [SVG tips](https://twitter.com/hashtag/SVGTipOfTheDay?src=hashtag_click\&f=live) - Louis Hoebregts
* [A Compendium of SVG Information](https://css-tricks.com/mega-list-svg-information/) - Chris Coyier
* [Making SVGs Responsive with CSS](https://tympanus.net/codrops/2014/08/19/making-svgs-responsive-with-css/) - Sara Soueidan
* [viewBox newsletter - archived](https://buttondown.email/viewBox/archive/) (SVG focus) - Cassie Evans and Louis Hoebregts
