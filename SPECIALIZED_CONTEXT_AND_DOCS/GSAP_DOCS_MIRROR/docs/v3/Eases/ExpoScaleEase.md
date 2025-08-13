# ExpoScaleEase

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(EasePack) 
```

#### Minimal usage

```
// we're starting at a scale of 1 and animating to 2, so pass those into config()...
gsap.to("#image", { duration: 1, scale: 2, ease: "expoScale(1, 2)" });
```

Not included in the Core

This ease is in the EasePack file. To learn how to include this in your project, see [the Installation page](/docs/v3/Installation).

### Description[​](#description "Direct link to Description")

There's an interesting phenomena that occurs when you animate an object's `scale` that makes it appear to change speed **even with a linear ease**; `ExpoScaleEase` compensates for this effect by bending the easing curve accordingly. This is the secret sauce for silky-smooth zooming/scaling animations.

### Video Explanation[​](#video-explanation "Direct link to Video Explanation")

Walkthrough

[YouTube video player](https://www.youtube.com/embed/rwdlO3uIlwk)

### Configuration[​](#configuration "Direct link to Configuration")

In order for ExpoScaleEase to create the correct easing curve, you must pass in the **starting** and **ending** scale values in the string, like:

```
// we're starting at a scale of 1 and animating to 2, so pass those into config()...
gsap.to("#image", { duration: 1, scale: 2, ease: "expoScale(1, 2)" });
```

It can also accept a 3rd parameter, the ease that you'd like it to bend (the default is `"none"`). So, for example, if you'd like to use `"power2.inOut"`, your code would look like:

```
//scale from 0.5 to 3 using "power2.inOut" ...
gsap.fromTo(
  "#image",
  { scale: 0.5 },
  { duration: 1, scale: 3, ease: "expoScale(0.5, 3, power2.inOut)" }
);
```

**Note:** The scale values passed into the `config()` method **must be non-zero** because the math wouldn't work with 0. You're welcome to use a small value like 0.01 instead. Using a *SUPER* small number like 0.00000001 may not be ideal because a large portion of the tween would be used going through the very small values.

### Simple Demo[​](#simple-demo "Direct link to Simple Demo")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/6239c068e182bdb8ff18926f519f8565?default-tab=result\&theme-id=41164)

### Complex Demo[​](#complex-demo "Direct link to Complex Demo")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/qBBBxaL?default-tab=result\&theme-id=41164)
