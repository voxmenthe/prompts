# updating to GSAP 3

info

Upgrading your project from using GSAP 2 to GSAP 3? It's easy. Most legacy code is already 100% compatible, but there are a few key differences to keep in mind. This guide will help move from GSAP 1.x/2.x to the all-new (and very exciting) GSAP 3.

## New Tween/Timeline Syntax (optional)[​](#new-tweentimeline-syntax-optional "Direct link to New Tween/Timeline Syntax (optional)")

The old syntax still works, but technically you never need to reference TweenMax, TweenLite, TimelineMax or TimelineLite anymore because they're all simplified into a single [`gsap`](/docs/v3/GSAP/.md) object!

```
// old 
TweenMax.to(".class", 2, {x: 100});

// new
gsap.to(".class", {duration: 2, x: 100});

// -- Timelines --

// old 
var tl = new TimelineMax();

// new
var tl = gsap.timeline();
```

Notice there's no "new" keyword needed to create the [`timeline`](/docs/v3/GSAP/Timeline.md).

Internally, there's one "[Tween](/docs/v3/GSAP/Tween.md)" class (replaces TweenLite/TweenMax) and one "[Timeline](/docs/v3/GSAP/Timeline.md)" class (replaces TimelineLite/TimelineMax), and both have ***all*** of the features like repeat, yoyo, etc. When you call one of the [gsap](/docs/v3/GSAP/.md) methods like [.to()](/docs/v3/GSAP/gsap.to\(\).md), [.from()](/docs/v3/GSAP/gsap.from\(\).md), etc., it returns an instance of the appropriate class with easily chainable methods. You never need to wonder which flavor (Lite/Max) you need.

So for the vast majority of your code, you could simply replace TweenLite and TweenMax with "gsap". You could also do a search/replace for `"new TimelineLite("` and `"new TimelineMax("`, replacing them with `"gsap.timeline("` (notice we left off the closing `")"` so that any vars objects are retained, like if you had `"new TimelineMax({repeat:-1})"` it'd keep the repeat).

## Duration (optional)[​](#duration-optional "Direct link to Duration (optional)")

You can still define a tween's duration as the 2nd parameter, but in GSAP 3 we encourage you to define it inside the vars object instead because it's more readable, it fits with the new keyframes feature, and can be function-based:

```
// old 
TweenMax.to(obj, 1.5, {...});
TweenMax.from(obj, 1.5, {...});
TweenMax.fromTo(obj, 1.5, {...}, {...});

// new
gsap.to(obj, {duration: 1.5, ...});
gsap.from(obj, {duration: 1.5, ...});
gsap.fromTo(obj, {...}, {duration: 1.5, ...});
```

## Easing (optional)[​](#easing-optional "Direct link to Easing (optional)")

The old eases still work great, but you can switch to the new, more compact ease format that requires less typing, is more readable, and eliminates import hassles. Simply include the ease name (all lowercase) followed by a dot and then the type (`".in"`, `".out"`, or `".inOut"`). Note that `.out` is the default so you can omit that completely.

```
// old
ease: Power3.easeInOut
ease: Sine.easeOut
ease: Linear.easeNone
ease: Elastic.easeOut.config(1, 0.5)
ease: SteppedEase.config(5);

// new
ease: "power3.inOut"
ease: "sine"  // the default is .out
ease: "none" // shortened keyword
ease: "elastic(1, 0.5)"
ease: "steps(5)"
```

Notice that for eases that support additional inputs, simply put them within some parenthesis at the end:

```
// old
ease: Elastic.easeOut.config(1, 0.3)
ease: Elastic.easeIn.config(1, 0.3)

// new
ease: "elastic(1, 0.3)"  // the default is .out
ease: "elastic.in(1, 0.3)"
```

[RoughEase](/docs/v3/Eases/RoughEase.md), [SlowMo](/docs/v3/Eases/SlowMo.md), and [ExpoScaleEase](/docs/v3/Eases/ExpoScaleEase.md) are not included in the core GSAP file - they're in an external EasePack file.

We highly recommend using our [Ease Visualizer](/docs/v3/Eases.md) to get the exact ease that you want and easily copy the correct formatting.

## Staggers (optional)[​](#staggers-optional "Direct link to Staggers (optional)")

The old stagger methods still exist for legacy code, but GSAP 3 supports staggers in ANY tween! Simply use the `stagger` property within the vars parameter.

```
// old
TweenMax.staggerTo(obj, 0.5, {...}, 0.1);

// new
// Simple stagger
gsap.to(obj, {..., stagger: 0.1});

// Complex stagger
gsap.to(obj, {..., stagger: {
  each: 0.1,
  from: "center"
  grid: "auto"
}});
```

warning

the old TweenMax.stagger\* methods returned an ***Array*** of tweens but the GSAP 3 legacy version returns a [Timeline](/docs/v3/GSAP/Timeline.md) instead. So if you have code that depends on an array being returned, you'll need to adjust your code. You can use [getChildren()](/docs/v3/GSAP/Timeline/getChildren\(\).md) method of the resulting timeline to get an array of nested tweens.

tip

**Handling repeats and onComplete**: if you add a repeat (like `repeat: -1`) to a staggered tween, it will wait until all the sub-tweens finish BEFORE repeating the entire sequence which can be quite handy but if you prefer to have each individual sub-tween repeat independently, just nest the `repeat` INSIDE the `stagger` object, like `stagger: {each: 0.1, repeat: -1}`. The same goes for `yoyo` and `onComplete`.

To learn more about staggers, check out [this article](https://codepen.io/GreenSock/pen/jdawKx).

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/jdawKx?default-tab=result\&theme-id=41164)

## Overwriting[​](#overwriting "Direct link to Overwriting")

Prior to GSAP 3, the default `overwrite` mode was `"auto"` which analyzes the tweens of the same target that are currently active/running and only overwrites individual properties that overlap/conflict, but in GSAP 3 the default mode is `false` meaning it won't check for any conflicts or apply any overwriting. Why? The overwrite behavior sometimes confused people, plus it required extra processing. We wanted to streamline things in GSAP 3 and make overwriting behavior an opt-in choice.

To get the GSAP 1.x/2.x behavior, simply do this once:

```
// set the default overwrite mode to "auto", like it was in GSAP 1.x/2.x
gsap.defaults({overwrite: "auto"});
```

Of course you can set overwrite on a per-tween basis too (in the vars object).

Also note that there were more overwrite modes in GSAP 1.x/2.x (like "concurrent", "preexisting" and "allOnStart") that have been eliminated in GSAP 3 to streamline things. Now the only options are `"auto"` (isolates only specific overlapping/conflicting properties), `false` (no overwriting), or `true` (when the tween starts, it immediately kills all other tweens of the same target regardless of which properties are being animated).

**onOverwrite** was removed in favor of a new **onInterrupt** callback that fires if/when the tween is killed before it completes. This could happen because its kill() method is called or due to overwriting.

## Plugins[​](#plugins "Direct link to Plugins")

### Loading plugins[​](#loading-plugins "Direct link to Loading plugins")

Similar to the old `TweenMax`, some plugins are already included in GSAP's core so that they don't need to be loaded separately. These are called [core plugins](/docs/v3/GSAP/CorePlugins) and include [AttrPlugin](/docs/v3/GSAP/CorePlugins/Attributes.md), [CSSPlugin](/docs/v3/GSAP/CorePlugins/CSS.md), [ModifiersPlugin](/docs/v3/GSAP/CorePlugins/Modifiers.md), and [SnapPlugin](/docs/v3/GSAP/CorePlugins/Snap.md). RoundPropsPlugin is also included for legacy code, but it has been replaced by the more flexible SnapPlugin.

Other plugins, such as [Draggable](/docs/v3/Plugins/Draggable/.md), [MotionPathPlugin](/docs/v3/Plugins/MotionPathPlugin.md), [MorphSVGPlugin](https://gsap.com/morphSVG), etc. need to be loaded separately and registered using [`gsap.registerPlugin()`](/docs/v3/GSAP/gsap.registerPlugin\(\).md). We recommend using the [GSAP Installation Helper](/docs/v3/Installation#install-helper) to get sample code showing how to load and register each file.

```
// register plugins (list as many as you'd like)
gsap.registerPlugin(MotionPathPlugin, TextPlugin, MorphSVGPlugin);
```

### MotionPathPlugin replaces BezierPlugin[​](#motionpathplugin-replaces-bezierplugin "Direct link to MotionPathPlugin replaces BezierPlugin")

GSAP's new [MotionPathPlugin](/docs/v3/Plugins/MotionPathPlugin.md) is essentially a better, more flexible version of the older BezierPlugin. In most cases, you can just change `bezier` legacy references to `motionPath`:

```
// old
bezier: [{x:200, y:100}, {x:400, y:0}, {x:300, y:200}]

// new
motionPath: [{x:200, y:100}, {x:400, y:0}, {x:300, y:200}]
```

Keep in mind that MotionPathPlugin also supports SVG paths! If you're having trouble converting your bezier curve to a motion path, feel free to post in [our forums](https://gsap.com/community/).

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/LwzMKL?default-tab=result\&theme-id=41164)

The old `type: "soft"` of BezierPlugin isn't available directly in MotionPathPlugin (it was rarely used), but there's a helper function in [this forums post](https://gsap.com/community/topic/24052-how-to-do-soft-bezier-with-motionpath/?tab=comments#comment-114049) that'll deliver identical results.

## className tweens removed[​](#classname-tweens-removed "Direct link to className tweens removed")

Support for class name tweens has been removed since they're not very performant, they're less clear, and required an uncomfortable amount of kb. Plus they were rarely used. Just use regular tweens instead that explicitly animate each property.

For example if you had this CSS:

```
.box {
  width: 100px;
  height: 100px;
  background-color: green;
}
.box.active {
  background-color: red;
}
```

You could use this JavaScript:

```
// old
.to(".class", 0.5, {className: "+=active"})

// new
.to(".class", {backgroundColor: "red"})

// if you need to add a class name in the end, you could do this instead:
.to(".class", {backgroundColor: "red", onComplete: function() {
  this.targets().forEach(elem => elem.classList.add("active"));
}})
```

## ColorPropsPlugin unnecessary[​](#colorpropsplugin-unnecessary "Direct link to ColorPropsPlugin unnecessary")

GSAP 3 has improved support for animating color values built into GSAP's core. As such, the old `ColorPropsPlugin` isn't necessary. Simply animate the color values directly as needed!

```
// old
TweenMax.to(myObject, 0.5, {colorProps: {borderColor: "rgb(204,51,0)"} });

// new
gsap.to(myObject, {borderColor: "rgb(204,51,0)", duration:0.5});
```

## skewType eliminated[​](#skewtype-eliminated "Direct link to skewType eliminated")

GSAP 3 removed `skewType` and `CSSPlugin.defaultSkewType` because they were rarely used and we wanted to conserve file size. If you still need this functionality, feel free to use the [`compensatedSkew` helper function](/docs/v3/HelperFunctions/helpers/compensated-skew).

## suffixMap[​](#suffixmap "Direct link to suffixMap")

`CSSPlugin.suffixMap` has been replaced by setting the `units` inside of [`gsap.config()`](/docs/v3/GSAP/gsap.config\(\).md) like:

```
// old
CSSPlugin.suffixMap.left = "%";

// new
gsap.config({units: {"left": "%"}})
```

## Cycle[​](#cycle "Direct link to Cycle")

GSAP 2.x stagger methods had a special `cycle` property that'd allow function-based values or arrays whose values would be cycled through, but GSAP 3 replaces this with a new even more flexible [gsap.utils.wrap()](/docs/v3/GSAP/UtilityMethods/wrap\(\).md) utility that can be used in ANY tween, not just staggers!

```
// old
TweenMax.staggerTo(".class", 0.5, {cycle: {x: [-100, 100]}}, 0.1)

// new
gsap.to(".class", {x: gsap.utils.wrap([-100, 100]), stagger: 0.1})
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/5364a46c2767c6258132f7805ea0035e?default-tab=result\&theme-id=41164)

## Ticker[​](#ticker "Direct link to Ticker")

If you want a function to run every time that GSAP updates (typically every requestAnimationFrame), simply add a listener to [`gsap.ticker`](/docs/v3/GSAP/gsap.ticker\(\)) with the new, simpler syntax:

```
// old
TweenLite.ticker.addEventListener("tick", myFunction);
TweenLite.ticker.removeEventListener("tick", myFunction);

// new
gsap.ticker.add(myFunction);
gsap.ticker.remove(myFunction);
```

Note that there is no `.useRAF()` function. GSAP 3 always uses `requestAnimationFrame` unless it is not supported, in which case it falls back to `setTimeout`.

## Defaults[​](#defaults "Direct link to Defaults")

Setting global defaults has been greatly simplified in GSAP 3. Instead of having static defaults (like `TweenLite.defaultEase`, `TweenLite.defaultOverwrite`, `CSSPlugin.defaultTransformPerspective`, and `CSSPlugin.defaultSmoothOrigin`), there is now one simple method where you can set all of these defaults: [`gsap.defaults()`](/docs/v3/GSAP/gsap.defaults\(\).md).

```
gsap.defaults({
  ease: "power2.in", 
  overwrite: "auto",
  smoothOrigin: false,
  transformPerspective: 500,
  duration: 1
});
```

You can also set defaults for each timeline instance which will be inherited by child tweens:

```
var tl = gsap.timeline({defaults: {
  ease: "power2.in", 
  duration: 1
} });

// now tweens created using tl.to(), tl.from(), and tl.fromTo() will use the
// above values as defaults
```

Other configuration values that aren't tween-specific can be set using [`gsap.config()`](/docs/v3/GSAP/gsap.config\(\).md) including what was formerly set using properties like `TweenLite.autoSleep` and `CSSPlugin.defaultForce3D`.

```
gsap.config({
  autoSleep: 60,
  force3D: false,
  nullTargetWarn: false,
  units: {left: "%", top: "%", rotation: "rad"}
});
```

## Callback scope[​](#callback-scope "Direct link to Callback scope")

In GSAP 3 scoping has been simplified. There is no more "scope" parameter in various methods like timeline's `call()` method, and no more `onUpdateScope`, `onStartScope`, `onCompleteScope`, or `onReverseCompleteScope`. Instead, use `callbackScope` to set the scope of *all* of the callback scopes of a particular tween/timeline or use `.bind` to set the scope of particular callbacks:

```
// old
TweenMax.to(obj, 0.5, {..., onCompleteScope: anotherObj, onComplete: function() {
  console.log(this); // logs anotherObj
}});

// new
gsap.to(obj, {..., callbackScope: anotherObj, onComplete: function() {
  console.log(this); // logs anotherObj
} });

// or 
gsap.to(obj, {..., onComplete: function() {
  console.log(this); // logs anotherObj
}.bind(anotherObj) });
```

You can access the tween itself by using `this` inside of the callback. In GSAP 1.x/2.x, you could reference a special `"{self}"` value in onCompleteParams, for example, but that's no longer valid because the callback is scoped to the tween instance itself by default. So, for example, you can get the tween's targets by using `this.targets()`. For example:

```
// old
TweenMax.to(obj, 0.5, {onComplete: function() {
  console.log(this.target);
}});


// new
gsap.to(obj, {onComplete: function() {
  console.log(this.targets()); // an array
}});
```

If `this.targets` is undefined, it's probably because you're using an [arrow function](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/Arrow_functions) which always locks its scope to where the arrow function was originally declared. If you want "this" to refer to the tween instance, just use a normal function instead of an arrow function.

```
gsap.to(".class", {
    // BE CAREFUL! Arrow functions lock scope to where they were created, so "this" won't refer to the tween instance here! 
    // Use normal functions if you need "this" to refer to the tween instance. 
    onComplete: () => console.log(this.targets()) // will not work
});
```

If you prefer using arrow functions (to lock scope to your object/context) and need to reference the tween instance in your callback, you could use this helper function:

```
// this function will always push the tween instance into the parameters for you and allow you to define a scope. 
function callback(func, scope, params) {
  let tween;
  params = params || [];
  return function() {
    if (!tween) {
      tween = this;
      params.push(tween);
    }
    func.apply(scope || tween, params);
  };
}
```

And then you could use it like this:

```
gsap.to(... {
  onComplete: callback(tween => {
    console.log(this); // since this is an arrow function, scope is locked anyway so this is your class instance
    console.log(tween); // tween instance
  })
});
```

## ThrowPropsPlugin renamed InertiaPlugin[​](#throwpropsplugin-renamed-inertiaplugin "Direct link to ThrowPropsPlugin renamed InertiaPlugin")

ThrowPropsPlugin has been renamed [InertiaPlugin](/docs/v3/Plugins/InertiaPlugin/.md) and has some new features.

## Other things to keep in mind[​](#other-things-to-keep-in-mind "Direct link to Other things to keep in mind")

### Transforms[​](#transforms "Direct link to Transforms")

We recommend setting all transform-related values via GSAP to maximize performance and avoid rotational and unit ambiguities. However, since it's relatively common for developers to set a value like `transform: translate(-50%, -50%)` in their CSS and the browser always reports those values in pixels, GSAP senses when the x/y translations are exactly -50% in pixels and sets `xPercent` or `yPercent` as a convenience in order to keep things centered. If you want to set things differently, again, just make sure you're doing so directly through GSAP, like `gsap.set("#id", {xPercent:0, x:100, yPercent:0, y:50})`.

### Getting an object's properties[​](#getting-an-objects-properties "Direct link to Getting an object's properties")

In GSAP 1.x/2.x, it was relatively common for developers to access an element's transform-specific properties via the undocumented \_gsTransform object but in GSAP 3 it's much easier. [`gsap.getProperty()`](/docs/v3/GSAP/gsap.getProperty\(\).md) lets you get **any** property, including transforms. There is no more `_gsTransform`.

```
// old
element._gsTransform.x

// new
gsap.getProperty(element, "x")
```

### Referring to the core classes[​](#referring-to-the-core-classes "Direct link to Referring to the core classes")

If you need to refer to the core [Tween](/docs/v3/GSAP/Tween.md) or [Timeline](/docs/v3/GSAP/Timeline.md) class, you can do so by referencing `gsap.core.Tween` and `gsap.core.Timeline`.

### timeScale() and reversed()[​](#timescale-and-reversed "Direct link to timeScale() and reversed()")

In GSAP 3 the timeScale controls the direction of playback, so setting it to a negative number makes the animation play backwards. That means it is intuitively linked with the `reversed()` method. If, for example, timeScale is 0.5 and then you call reverse() it will be set to -0.5. In GSAP 2 and earlier, the "reversed" state of the animation was completely independent from timeScale (which wasn't allowed to be negative). So in GSAP 3, you could even animate timeScale from positive to negative and back again!

## Removed methods/properties[​](#removed-methodsproperties "Direct link to Removed methods/properties")

* **TweenLite.selector** - There's no more `TweenLite.selector` or `TweenMax.selector` (it's pointless with `document.querySelectorAll()` that's in browsers now).

* **timeline.addCallback()** - dropped in favor of the simpler `.call()` method.

* **TweenMax's pauseAll(), resumeAll(), killAll(), and globalTimeScale()** - dropped in favor of directly accessing methods on the [`globalTimeline`](/docs/v3/GSAP/gsap.globalTimeline\(\)), like:

  <!-- -->

  ```
  gsap.globalTimeline.pause();
  gsap.globalTimeline.resume();
  gsap.globalTimeline.clear(); // like killAll()
  gsap.globalTimeline.timeScale(0.5);
  ```

## Frequently Asked Questions (FAQ)[​](#frequently-asked-questions-faq "Direct link to Frequently Asked Questions (FAQ)")

#### Why migrate to GSAP 3?

GSAP 3 is almost half the file size of the old TweenMax, has 50+ more features, and has a simpler API.

#### Do I have to use the new syntax?

We highly recommend that you use the new syntax, but no, it's not imperative. Most old GSAP syntax will work just fine in GSAP 3. We're pretty confident that you'll love the new syntax once you're used to it!

#### Will GSAP 2.x be actively maintained for years?

We'll certainly answer questions in [the forums](https://gsap.com/community/) and help users of GSAP 2.x, but we're focusing all of our development resources on the more modern 3.x moving forward so don't expect any additional 2.x releases in the future.

#### My production build isn't working with GSAP 3. Why?

Usually this just means that your build tool is applying tree shaking and dumping plugins - that's why you need to register your plugins with [gsap.registerPlugin()](/docs/v3/GSAP/gsap.registerPlugin\(\).md). We recommend that you use the [Installation Helper](/docs/v3/Installation#install-helper) which gives you code for proper registration as well.

#### I am seeing some odd/unexpected behavior but don't have any errors. What's going on?

Try setting `gsap.defaults({overwrite: "auto"})` and see if that fixes the issue. If it does, you must have created some conflicting tweens. You could either keep the default overwrite value of `"auto"` or restructure your animation to avoid the conflict.

If that doesn't fix the issue, please post in [our forums](https://gsap.com/community/) and we'd be happy to help!

## More information[​](#more-information "Direct link to More information")

For a deep dive into the nitty-gritty of GSAP 3, check out the [GSAP 3 Release Notes](/blog/3-release-notes). As always, if you have questions or are having trouble [our forums](https://gsap.com/community/) are available to help you!
