# RoughEase

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

Most easing equations give a smooth, gradual transition between the start and end values, but `RoughEase` provides an easy way to get a rough, jagged effect instead, or you can also get an evenly-spaced back-and-forth movement if you prefer. `RoughEase` is in the EasePack file. Configure the `RoughEase` with any of these optional properties:

### Config Object[​](#config-object "Direct link to Config Object")

* ### Property

  ### Description

  #### clamp[](#clamp)

  Boolean - Setting `clamp` to `true` will prevent points from exceeding the end value or dropping below the starting value. For example, if you’re tweening the x property from 0 to 100, the RoughEase would force all random points to stay between 0 and 100 if `clamp` is `true`, but if it is `false`, x could potentially jump above 100 or below 0 at some point during the tween (it would always end at 100 though in this example). Default: `false`.

* #### points[](#points)

  Number - The number of points to be plotted along the ease, making it jerk more or less frequently. Default: `20`.

* #### randomize[](#randomize)

  Boolean - By default, the placement of points will be randomized (creating the roughness) but you can set `randomize` to `false` to make the points zig-zag evenly across the ease. Using this in conjunction with a `taper` value can create a nice effect. Default: `true`.

* #### strength[](#strength)

  Number - Controls how far from the template ease the points are allowed to wander (a small number like 0.1 keeps it very close to the template ease whereas a larger number like 5 creates much bigger variations). Default: `1`.

* #### taper[](#taper)

  String (`"in"` | `"out"` | `"both"` | `"none"`) - To make the strength of the roughness taper towards the end or beginning or both, use `"out"`, `"in"`, or `"both"` respectively. Default: `"none"`.

* #### template[](#template)

  String - An ease that should be used as a template, like a general guide. The RoughEase will plot points that wander from that template. You can use this to influence the general shape of the RoughEase. Default: `"none"`.

### Example code[​](#example-code "Direct link to Example code")

```
//use the default values
gsap.from(element, {duration: 1, opacity: 0, ease: "rough"});

//or customize the configuration
gsap.to(element, {duration: 2, y: 300, ease: "rough({strength: 3, points: 50, template: strong.inOut, taper: both, randomize: false})" });
```
