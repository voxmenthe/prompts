# Inertia

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(InertiaPlugin) 
```

#### Minimal usage

```
gsap.to(obj, { inertia: { x: 500, y: -300 } });
```

## Description[​](#description "Direct link to Description")

InertiaPlugin (formerly ThrowPropsPlugin) allows you to smoothly glide any property to a stop, honoring an initial velocity as well as applying optional restrictions on the end value. You can define a specific end value or allow it to be chosen automatically based on the initial velocity (and ease) or you can define a max/min range or even an array of snap-to values that act as notches. You can have it "watch" certain properties to keep track of their velocities and then use them automatically when you do an `inertia` tween. This is perfect for flick-scrolling or animating things as though they are being thrown (where momentum factors into the animation).

For example, let's say a user drags a ball and and then when released, the ball should continue flying at the same velocity as it was just moving (so that it appears seamless), and then glide to a rest. You can't do a normal tween because you don't know exactly where it should land or how long the tween should last (faster initial velocity would usually mean a longer duration). You'd like it to decelerate based on whatever ease you define in your tween.

Maybe you want the final resting value to always land within a particular range so that the ball doesn't fly off the edge of the screen. But you don't want it to suddenly jerk to a stop when it hits the edge of the screen either; instead, you want it to ease gently into place even if that means going past the landing spot briefly and easing back (if the initial velocity is fast enough to require that). The whole point is to make it look **smooth**.

**No problem.**

In its simplest form, you can pass just the initial velocity for each property like this:

```
gsap.to(obj, { inertia: { x: 500, y: -300 } });
```

In the above example, `obj.x` will animate at 500 pixels per second initially and `obj.y` will animate at -300 pixels per second. Both will decelerate smoothly until they come to rest and InertiaPlugin figures out a natural-looking duration for you!

To impose maximum and minimum boundaries on the end values, use the object syntax with the `max` and `min` special properties like this:

```
gsap.to(obj, {
  inertia: {
    x: {
      velocity: 500,
      max: 1024,
      min: 0,
    },
    y: {
      velocity: -300,
      max: 720,
      min: 0,
    },
  },
});
```

Notice the nesting of the objects (`{}`). The `max` and `min` values refer to the range for the final resting position (coordinates in this case), **not** the velocity. So `obj.x` would always land between 0 and 1024 in this case, and `obj.y` would always land between 0 and 720. If you want the target object to land on a specific value rather than within a range, simply set `max` and `min` to identical values or just use the `end` property. Also notice that you must define a `velocity` value for each property (unless you're using `track()` - see below for details).

## Config Object[​](#config-object "Direct link to Config Object")

* ### Property

  ### Description

  #### velocity[](#velocity)

  \[*Number* | *“auto”*] - The initial velocity, measured in units per second. You may omit velocity or just use “auto” for properties that are being tracked automatically using the track() method.

* #### min[](#min)

  Number - The minimum end value of the property. For example, if you don’t want `x` to land at a value below 0, your `inertia` may look like `{x: {velocity: -500, min: 0}}`.

* #### max[](#max)

  Number - The maximum end value of the property. For example, if you don’t want `x` to exceed 1024, your `inertia` may look like `{x: {velocity: 500, max: 1024}}`.

* #### end[](#end)

  \[*Number* | *Array* | *Function*] - If `end` is defined as a **Number**, the target will land EXACTLY there (just as if you set both the `max` and `min` to identical values). If `end` is defined as a numeric **Array**, those values will be treated like “notches” or “snap-to” values so that the closest one to the natural landing spot will be selected. For example, if `[0,100,200]` is used, and the value would have naturally landed at 141, it will use the closest number (100 in this case) and land there instead. If end is defined as a **Function**, that function will be called and passed the natural landing value as the only parameter, and your function can run whatever logic you want, and then return the number at which it should land. This can be useful if, for example, you have a rotational tween and you want it to snap to 10-degree increments no matter how big or small, you could use a function that just rounds the natural value to the closest 10-degree increment. So any of these are valid: `end: 100`, `end: [0,100,200,300]`, or `end: function(n) { return Math.round(n / 10) * 10; }`.

* #### linkedProps[](#linkedProps)

  String - A comma-delimited list of properties that should be linked together into a single object when passed to a function-based `end` value so that they’re processed together. This is only useful when, for example, you have an `x` and `y` but the logic in your end function needs BOTH of those (like for snapping coordinates). See [this demo](//codepen.io/GreenSock/pen/aqEdGM?editors=0010) for an example. The object that gets passed as the only parameter to the `end` function will have the properties are listed in `linkedProps`. So, for example, if `linkedProps` is `"x,y"`, then an object like `{x: 100, y: 140}` gets passed to the function as a parameter. Those values are the natural ending values, but of course your function should return a similar object with the new values you want the end values to be, like `return {x: 200, y: 300}`.

* #### resistance[](#resistance)

  Number - The amount of resistance per second (think of it like how much friction is applied)..

InertiaPlugin isn't just for tweening x and y coordinates. It works with any numeric property, so you could use it for spinning the `rotation` of an object as well. Or the `scaleX`/`scaleY` properties. Maybe the user drags to spin a wheel and lets go and you want it to continue increasing the `rotation` at that velocity, decelerating smoothly until it stops. It even works with method-based getters/setters.

## Automatically determine duration[​](#automatically-determine-duration "Direct link to Automatically determine duration")

One of the trickiest parts of creating a `inertia` tween that looks fluid and natural (particularly if you're applying maximum and/or minimum values) is determining its duration. Typically it's best to have a relatively consistent level of resistance so that if the initial velocity is very fast, it takes longer for the object to come to rest compared to when the initial velocity is slower. You also may want to impose some restrictions on how long a tween can last (if the user drags incredibly fast, you might not want the tween to last 200 seconds). The duration will also affect how far past a max/min boundary the property may go, so you might want to only allow a certain amount of overshoot tolerance. That's why InertiaPlugin automatically sets the duration of the Tween for you, and you can optionally hard-code a duration in the inertia object or even use max/min values to give it a range, like `duration:{min:0.5, max:3}`.

## Automatically track velocity[​](#automatically-track-velocity "Direct link to Automatically track velocity")

Another tricky aspect of smoothly transitioning from a particular velocity is tracking the property's velocity in the first place! So we've made that easier too - you can use the [`InertiaPlugin.track()`](/docs/v3/Plugins/InertiaPlugin/.md) method to have the velocity (rate of change) of certain properties tracked and then `inertia` tweens will automatically grab the appropriate tracked value internally, allowing you to omit the `velocity` values in your tweens altogether. See the [`track()`](/docs/v3/Plugins/InertiaPlugin/static.track\(\).md) method's description for details. And make sure you start tracking velocity at least a half-second before you need to tween because it takes a small amount of time to gauge how fast something is going.

A unique convenience of InertiaPlugin compared to most other solutions out there that use frame-based loops is that everything is reversible and you can jump to any spot in the tween immediately. So if you create several `inertia` tweens, for example, and dump them into a timeline, you could simply call `reverse()` on the timeline to watch the objects retrace their steps right back to the beginning!

## Examples[​](#examples "Direct link to Examples")

The following example creates a green box and a red box that you can drag and toss around the screen in a natural, fluid way. If you check the "Snap to grid" checkbox, the boxes will always land exactly on the grid. We use `Draggable` class so that we can focus more on the InertiaPlugin code rather than all the boilerplate code needed to make things draggable:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/JjwZzNG?default-tab=result\&theme-id=41164)

The following example demonstrates using a custom `end` function for complex snapping that requires both x and y values, thus `linkedProps` is used:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/qBBWPPe?default-tab=result\&theme-id=41164)

Although InertiaPlugin is commonly used with Draggable, Draggable is not required an InertiaPlugin can be used independently from Draggable.. Here's an example of Inertia's tracking being used directly:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/myyJrLE?default-tab=result\&theme-id=41164)

## **Methods**[​](#methods "Direct link to methods")

|                                                                                                                                                 |                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #### [InertiaPlugin.getVelocity](/docs/v3/Plugins/InertiaPlugin/static.getVelocity\(\).md)( target:Element \| String, property:String ) ;       | Returns the current velocity of the given property and target object (only works if you started tracking the property using the [`InertiaPlugin.track()`](/docs/v3/Plugins/InertiaPlugin/static.track\(\).md) method). |
| #### [InertiaPlugin.isTracking](/docs/v3/Plugins/InertiaPlugin/static.isTracking\(\).md)( target:Element \| String, property:String ) : Boolean |                                                                                                                                                                                                                        |
| #### [InertiaPlugin.track](/docs/v3/Plugins/InertiaPlugin/static.track\(\).md)( target:Element \| String \| Array, props:String ) : Array       |                                                                                                                                                                                                                        |
| #### [InertiaPlugin.untrack](/docs/v3/Plugins/InertiaPlugin/static.untrack\(\).md)( target:Element \| String \| Array, props:String ) ;         |                                                                                                                                                                                                                        |

## **Demos**[​](#demos "Direct link to demos")

Inertia pairs great with Draggable - check out the Draggable [how-to collection](https://codepen.io/collection/AtuHb) and our favourite [inspiring community demos](https://codepen.io/collection/DrQGpM) on CodePen.

Inertia Demos

Search..

\[x]All

Play Demo videos\[ ]

