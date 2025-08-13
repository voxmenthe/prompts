# Physics2D

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(Physics2DPlugin) 
```

#### Minimal usage

```
gsap.to(element, {
  duration: 2,
  physics2D: { velocity: 300, angle: -60, gravity: 400 },
});
```

## Description[​](#description "Direct link to Description")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/YzdOVbK?default-tab=result\&theme-id=41164)

Provides simple physics functionality for tweening an Object's `x` and `y` coordinates (or `left` and `top`) based on a combination of `velocity`, `angle`, `gravity`, `acceleration`, `accelerationAngle`, and/or `friction`. It is not intended to replace a full-blown physics engine and does not offer collision detection, but serves as a way to easily create interesting physics-based effects with the GreenSock Animation Platform.

## **Config Object**[​](#config-object "Direct link to config-object")

* ### Property

  ### Description

  #### velocity[](#velocity)

  Number - The initial velocity of the object measured in pixels per time unit. Default: `0`.

* #### angle[](#angle)

  Number - The initial angle (in degrees) at which the object should travel. This only matters when a `velocity` is defined. For example, if the object should start out traveling at -60 degrees (towards the upper right), the `angle` would be `-60`. Default: `0`.

* #### gravity[](#gravity)

  Number - The amount of downward acceleration applied to the object, measured in pixels per second. You can either use `gravity` or `acceleration`, not both because `gravity` is the same thing as `acceleration` applied at an `accelerationAngle` of `90`. Think of gravity as a convenience property that automatically sets the `accelerationAngle` for you. Default: `null`.

* #### acceleration[](#acceleration)

  Number - The amount of acceleration applied to the object, measured in pixels per second, it would be measured per frame). To apply the acceleration in a specific direction that is different than the angle, use the `accelerationAngle` property. You can either use `gravity` or `acceleration`, not both because `gravity` is the same thing as `acceleration` applied at an `accelerationAngle` of `90`. Default: `null`.

* #### accelerationAngle[](#accelerationAngle)

  Number - The angle at which acceleration is applied (if any), measured in degrees. So if, for example, you want the object to accelerate towards the left side of the screen, you’d use an `accelerationAngle` of `180`. If you define a `gravity` value, it will automatically set the `accelerationAngle` to `90` (downward). Default: `null`.

* #### friction[](#friction)

  Number - A value between 0 and 1 where 0 is no friction, 0.08 is a small amount of friction, and 1 will completely prevent any movement. This is not meant to be precise or scientific in any way, but it serves as an easy way to apply a friction-like physics effect to your tween. Generally it is best to experiment with this number a bit - start with very small values like 0.02. Also note that friction requires more processing than physics tweens without any friction. Default: `0`.

* #### xProp[](#xProp)

  String - By default, the `x` property of the target object is used to control x-axis movement, but if you’d prefer to use a different property name, use `xProp` like `xProp: "left"`. Default: `"x"`.

* #### yProp[](#yProp)

  String - By default, the `y` property of the target object is used to control y-axis movement, but if you’d prefer to use a different property name, use `yProp` like `yProp: "top"`. Default: `"y"`.

info

Parameters are not intended to be dynamically updateable, but one unique convenience is that everything is reverseable. So if you spawn a bunch of particle tweens, for example, and throw them into a timeline, you could simply call `reverse()` on the timeline to watch the particles retrace their steps right back to the beginning. Keep in mind that any easing equation you define for your tween will be completely ignored for these properties.

## Usage[​](#usage "Direct link to Usage")

```
gsap.to(element, {
  duration: 2,
  physics2D: { velocity: 300, angle: -60, gravity: 400 },
});
//or
gsap.to(element, {
  duration: 2,
  physics2D: { velocity: 300, angle: -60, friction: 0.1 },
});
//or
gsap.to(element, {
  duration: 2,
  physics2D: {
    velocity: 300,
    angle: -60,
    acceleration: 50,
    accelerationAngle: 180,
  },
});
```

## **Demos**[​](#demos "Direct link to demos")

Check out more [Physics2D demos on codepen](https://codepen.io/collection/AORzWx)

Physics2D Demos

Search..

\[x]All

Play Demo videos\[ ]

