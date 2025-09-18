# Pixi

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(PixiPlugin) 
```

#### Minimal usage

```
 gsap.to(graphics, { duration: 2, pixi: { lineColor: "purple" } });
```

PixiPlugin makes it much easier to animate things in [PixiJS](//www.pixijs.com/), a popular canvas library that's extremely performant. Without the plugin, it's a tad cumbersome with certain properties because they're tucked inside sub-objects in PixiJS's API, like `object.position.x`, `object.scale.y`, `object.skew.x`, etc. Plus PixiJS defines rotational values in radians instead of degrees which isn't as intuitive for most developers and designers. PixiPlugin saves you a bunch of headaches:

```
//old way (without plugin):
gsap.to(pixiObject.scale, { x: 2, y: 1.5, duration: 1 });
gsap.to(pixiObject.skew, { x: (30 * Math.PI) / 180, duration: 1 });
gsap.to(pixiObject, { rotation: (60 * Math.PI) / 180, duration: 1 });

//new way (with plugin):
gsap.to(pixiObject, {
  pixi: { scaleX: 2, scaleY: 1.5, skewX: 30, rotation: 60 },
  duration: 1,
});
```

Notice **rotational values are defined in degrees, not radians**. Yay!

Be sure to include the PixiPlugin correctly:

```
import * as PIXI from "pixi.js";
import { gsap } from "gsap";
import { PixiPlugin } from "gsap/PixiPlugin";

// register the plugin
gsap.registerPlugin(PixiPlugin);

// give the plugin a reference to the PIXI object
PixiPlugin.registerPIXI(PIXI);
```

## PixiJS examples[​](#pixijs-examples "Direct link to PixiJS examples")

There are a bunch of GSAP-based examples in the [PixiJS documentation here](https://pixijs.io/examples/#/gsap3-interaction/gsap3-basic.js)! It's a great place to start.

## Colors[​](#colors "Direct link to Colors")

PixiJS requires that you define color-related values in a format like `0xFF0000` but with PixiPlugin, you can define them the same way you would in CSS, like `"red"`, `"#F00"`, `"#FF0000"`, `"rgb(255,0,0)"`, `"hsl(0, 100%, 50%)"`, or `0xFF0000`. You can even do relative HSL values! `"hsl(+=180, +=0%, +=0%)"`.

```
//named colors
gsap.to(graphics, { duration: 2, pixi: { lineColor: "purple" } });

//relative hsl() color that reduces brightness but leaves the hue and saturation the same:
gsap.to(graphics, {
  duration: 2,
  pixi: { fillColor: "hsl(+=0, +=0%, -=30%)" },
});
```

## ColorMatrixFilter[​](#colormatrixfilter "Direct link to ColorMatrixFilter")

Another big convenience is that PixiPlugin recognizes some special values like `saturation`, `brightness`, `contrast`, `hue`, and `colorize` (which all leverage a `ColorMatrixFilter` under the hood).

```
var image = new PIXI.Sprite.from(
  "http://pixijs.github.io/examples/required/assets/panda.png"
);
app.stage.addChild(image);

var tl = gsap.timeline({ defaults: { duration: 2 } });
//colorize fully red. Change colorAmount to 0.5 to make it only halfway colorized, for example:
tl.to(image, { pixi: { colorize: "red", colorizeAmount: 1 } })
  //change the hue 180 degrees (opposite)
  .to(image, { pixi: { hue: 180 } })
  //completely desaturate
  .to(image, { pixi: { saturation: 0 } })
  //blow out the brightness to double the normal amount
  .to(image, { pixi: { brightness: 2 } })
  //increase the contrast
  .to(image, { pixi: { contrast: 1.5 } });
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/e1f256723ce102d6b9a776fa7f6da9f4?default-tab=result\&theme-id=41164)

Or if you have a custom `ColorMatrixFilter`, just pass that in as the `colorMatrixFilter` property and it'll handle animating between states:

```
var filter = new PIXI.filters.ColorMatrixFilter();
filter.sepia();
gsap.to(image, { pixi: { colorMatrixFilter: filter }, duration: 2 });
```

## BlurFilter[​](#blurfilter "Direct link to BlurFilter")

PixiPlugin recognizes `blur`, `blurX`, and `blurY` properties, so it's very simple to apply a blur without having to create a new `BlurFilter` instance, add it to the filters array, and animate its properties separately.

```
//blur on both the x and y axis to a blur amount of 15
gsap.to(image, { pixi: { blurX: 15, blurY: 15 }, duration: 2 });
```

## Directional rotation[​](#directional-rotation "Direct link to Directional rotation")

You can control which direction a rotation tween goes by appending a suffix for **clockwise** (`"_cw"`), **counter-clockwise** (`"_ccw"`), or the **shortest direction** (`"_short"`). For example, if the element's rotation is currently 170 degrees and you want to tween it to -170 degrees, a normal rotation tween would travel a total of 340 degrees in the counter-clockwise direction, but `rotation: "-170_short"` suffix, it would travel 20 degrees in the clockwise direction instead! Example:

```
gsap.to(element, {
  pixi: { rotation: "-170_short" },
  duration: 2,
});
```

Directional rotation capabilities were added in GSAP 3.2, so make sure you've got the latest update.

## Other properties[​](#other-properties "Direct link to Other properties")

PixiPlugin can handle almost any other property as well - there is no pre-determined list of "allowed" properties. PixiPlugin simply improves developer ergonomics for anyone animating in PixiJS. Less code, fewer headaches, and faster production. For a full listing of properties that the PixiPlugin helps with, see [the PixiPlugin Typescript declarations](https://github.com/greensock/GSAP/blob/master/types/pixi-plugin.d.ts).

## **Methods**[​](#methods "Direct link to methods")

|                                                                                                         |                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| #### [PixiPlugin.registerPIXI](/docs/v3/Plugins/PixiPlugin/static.registerPIXI\(\).md)( PIXI:Object ) ; | Registers the main PIXI library object with the PixiPlugin so that it can find the necessary classes/objects. You only need to register it once. |
