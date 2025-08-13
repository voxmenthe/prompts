# Animating backgroundSize:"cover" or "contain"

I was asked about animating to or from a `backgroundSize` of `"cover"` or `"contain"` with GSAP. ﻿**The problem:** GSAP interpolates between numbers, but how is it supposed to interpolate between something like "300px 250px" and "contain" (not a number)? So I whipped together a function that basically translates "contain" or "cover" into their px-based equivalents for that particular element at whatever size it is then. Once we've got it converted, it's easy t﻿o animate.

```
/*
config is optional and can have any of the following properties:
- size [string] - the size to set and convert into px before it gets returned, like "cover" or "150% auto".
- nativeWidth [number] - native width of the image (in pixels)
- nativeHeight [number] - native height of the image (in pixels)

Simple example:
// returns current backgroundSize in px
bgSize(".class");

Advanced example:
// sets the backgroundSize to "cover" and returns it in the equivalent px-based amount assuming the image's native width is 600px and height is 400px.
bgSize(".class", {size: "cover", nativeWidth: 600, nativeHeight: 400});

Note: if you can define the nativeWidth and nativeHeight, it helps becaues it can skip tasks like creating
an Image and loading the URL to detect the native size automatically. Sometimes images don't load fast enough,
so skipping that step avoids the whole issue.
*/
function bgSize(element, config) {
  config = config || {};
  let e = gsap.utils.toArray(element)[0],
    cs = window.getComputedStyle(e),
    imageUrl = cs.backgroundImage,
    { nativeWidth, nativeHeight } = config,
    size = config.size || cs.backgroundSize,
    image,
    w,
    h,
    ew,
    eh,
    ratio;
  if (imageUrl && (!/\d/g.test(size) || size.indexOf("%") > -1)) {
    if (!nativeWidth || !nativeHeight) {
      image = new Image();
      image.setAttribute(
        "src",
        imageUrl.replace(/(^url\("|^url\('|^url\(|"\)$|'\)$|\)$)/gi, "")
      );
      nativeWidth = image.naturalWidth;
      nativeHeight = image.naturalHeight;
    }
    ew = e.offsetWidth;
    eh = e.offsetHeight;
    if (!nativeWidth || !nativeHeight) {
      console.log("bgSize() failed;", imageUrl, "hasn't loaded yet.");
      nativeWidth = ew;
      nativeHeight = eh;
    }
    ratio = nativeWidth / nativeHeight;
    if (size === "cover" || size === "contain") {
      if ((size === "cover") === nativeWidth / ew > nativeHeight / eh) {
        h = eh;
        w = eh * ratio;
      } else {
        w = ew;
        h = ew / ratio;
      }
    } else {
      // "auto" or %
      size = size.split(" ");
      size.push("");
      w = ~size[0].indexOf("%")
        ? (ew * parseFloat(size[0])) / 100
        : nativeWidth;
      h = ~size[1].indexOf("%")
        ? (eh * parseFloat(size[1])) / 100
        : nativeHeight;
    }
    size = Math.ceil(w) + "px " + Math.ceil(h) + "px";
    config.size && (e.style.backgroundSize = size);
  }
  return size;
}
```

## Demo[​](#demo "Direct link to Demo")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/89bd92e9114108e96874df09a424e8aa?default-tab=result\&theme-id=41164)

## BackgroundSizePlugin[​](#backgroundsizeplugin "Direct link to BackgroundSizePlugin")

For even more flexibility, you can use this unofficial plugin for animating the backgroundSize to/from "cover" or "contain" and it'll even let you apply a scale to the value. Plus if you animate to "cover" or "contain", it will actually set it to that value (instead of the pixel-based equivalent) so that it's responsive after the tween:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/rNYxENg?default-tab=result\&theme-id=41164)
