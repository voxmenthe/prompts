# Easel

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(EaselPlugin) 
```

#### Minimal usage

```
gsap.ticker.add(() => stage.update());

gsap.to(circle, {
  duration: 2,
  scaleX: 0.5,
  scaleY: 0.5,
  easel: { tint: 0x00ff00 },
});
```

### Description[​](#description "Direct link to Description")

Tweens special EaselJS-related properties for things like `saturation`, `contrast`, `tint`, `colorize`, `brightness`, `exposure`, and `hue` which leverage EaselJS's `ColorFilter` and `ColorMatrixFilter` (see [the EaselJS website](//www.createjs.com/#!/EaselJS) for more information about EaselJS). You don't need the plugin to tween normal numeric properties of EaselJS objects (like `x` and `y`), but some filters or effects require special manipulation which is what EaselPlugin is for. Currently it only handles special properties related to `ColorFilter` and `ColorMatrixFilter`, and it can tween the `frame` property of a `MovieClip`.

GreenSock's EaselPlugin exposes convenient properties that aren't a part of EaselJS's API like `tint`, `tintAmount`, `exposure`, and `brightness` for `ColorFilter`, as well as `saturation`, `hue`, `contrast`, `colorize`, and `colorizeAmount` for `ColorMatrixFilter`. Simply wrap the values that you'd like to tween in an `easel: {}` object. Here are some examples:

```
//setup stage and create a Shape into which we'll draw a circle later...
var canvas = document.getElementById("myCanvas"),
  stage = new createjs.Stage(canvas),
  circle = new createjs.Shape(),
  g = circle.graphics;

//draw a red circle in the Shape
g.beginFill(createjs.Graphics.getRGB(255, 0, 0));
g.drawCircle(0, 0, 100);
g.endFill();

//in order for the ColorFilter to work, we must cache() the circle
circle.cache(-100, -100, 200, 200);

//place the circle at 200,200
circle.x = 200;
circle.y = 200;

//add the circle to the stage
stage.addChild(circle);

//setup a "tick" event listener so that the EaselJS stage gets updated on every frame/tick
gsap.ticker.add(() => stage.update());
stage.update();

//tween the tint of the circle to green and scale it to half-size
gsap.to(circle, {
  duration: 2,
  scaleX: 0.5,
  scaleY: 0.5,
  easel: { tint: 0x00ff00 },
});

//tween to a different tint that is only 50% (mixing with half of the original color) and animate the scale, position, and rotation simultaneously.
gsap.to(circle, {
  duration: 3,
  scaleX: 1.5,
  scaleY: 0.8,
  x: 250,
  y: 150,
  rotation: 180,
  easel: { tint: "#0000FF", tintAmount: 0.5 },
  delay: 3,
  ease: "elastic",
});

//then animate the saturation down to 0
gsap.to(circle, { duration: 2, easel: { saturation: 0 }, delay: 6 });
```

You can also tween any individual properties of the `ColorFilter` object like this:

```
gsap.to(circle, {
  duration: 3,
  easel: {
    colorFilter: { redMultiplier: 0.5, blueMultiplier: 0.8, greenOffset: 100 },
  },
});
```

Or you can tween things like the `exposure` of an image which is a value from 0-2 where 1 is normal exposure, 2 is completely overexposed (white) and 0 is completely underexposed (black). Or define a `brightness` value which uses the same concept: a value from 0-2. These effects can be very useful for images in particular.

note

A common mistake is to forget to wrap EaselJS-related properties in an easel object which is essential for specifying your intent. You also must load the EaselJS's `ColorFilter` and/or `ColorMatrixFilter` JavaScript files.
