# Common GSAP mistakes

info

With over 180,000 posts in the GSAP Forums, weve noticed some common mistakes that you 'd be wise to avoid. We threw in a few tips as well.

## Creating from() logic issues[​](#creating-from-logic-issues "Direct link to Creating from() logic issues")

It 's usually smart to use `.to()` and `.from()` tweens instead of `.fromTo()` because they 're more dynamic - they pull either the starting or ending values from whatever they happen to *CURRENTLY* be at the time that tween renders for the first time. It's one of the tips in the article on [animating efficiently](https://css-tricks.com/tips-for-writing-animation-code-efficiently/). But be careful because that dynamic nature can bite you in a few scenarios.

First, keep in mind that `.from()` tweens go from the provided value to the *current* value. Take a look at this example:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/gOPXLwN?default-tab=result\&theme-id=41164)

Try clicking it one time and letting it play. It works, fading in the element.

Now try clicking it multiple times right after each other. The box stops showing up because it uses the current opacity as the end point which, if the animation has not completed, is some value less than 1.

The fix for this is simple: use a `.fromTo()`. Alternatively you could create the animation beforehand and use a control method (we 'll talk more about this approach later in this article).

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/vYLWyWw?default-tab=result\&theme-id=41164)

Second, keep in mind that by default `immediateRender` is true by default for `.from()` and `.fromTo()` tweens because that 's typically the most intuitive behavior (if you 're animating from a certain value, it should start there right away). But if you create a `.from()` tween **after** a `.to()` tween affecting the *same* properties of the *same* object, try to figure out what will happen:

```
const tl = gsap.timeline()
tl.to(".box", {x: 100});
tl.from(".box", {x: 100});
```

You might expect the box to animate x from 0 to 100 and then back to 0. Or maybe you 'd expect it to animate from 0 to 100 and then stay at 100. Let's see what happens:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/dyGOjGQ?default-tab=result\&theme-id=41164)

The box animates x from 100 to 100 and then back to 0. Why is that?

By default `.to()` tweens wait to render until their playhead actually moves (it 's a waste of CPU cycles to render at a time of 0 because nothing will have changed). But since `from()` has `immediateRender: true`, `x` jumps to 100 immediately on the current tick! *Then* it runs the `.to()` tween on the next tick (since it's first in the timeline) and records the current starting value which is 100! So it animates 100 to 100 over 0.5 seconds. Then it runs the `.from()` tween which has the cached value of 0 as the end value.

If you have several timelines affecting the same element, situations like this can be a little tricky to catch. So just be mindful of how things work when using `.to()` and `.from()` tweens. They're very powerful but with power comes responsibility. A simple solution here is to set `immediateRender: true` on the `.to()` tween, or `immediateRender: false` on the `.from()` tween.

The third situation is similar but involves [repeatRefresh](/3-1/#repeatRefresh-tl) and repeats.

Let's say you have a situation where you want a looped animation that fades in some text and fades it out. You could create a timeline, use a `.from()` to fade in the text, then use a `.to()` to fade it out:

```
const tl = gsap.timeline({repeat:-1});
tl.set(".text", { color: "random([green, gray, orange, pink])" }, 2);	
tl.from(chars, { opacity: 0 });
tl.to(chars, { opacity: 0 });
```

This will work just fine! Here's the same thing but staggered using SplitText to make it look a little nicer:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ZEQBgNg?default-tab=result\&theme-id=41164)

But this only randomizes the colors at the start. What if we want new random values each repeat? That's where `repeatRefresh` comes in. Let's add `repeatRefresh: true` to see what happens:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/KKVNOYJ?default-tab=result\&theme-id=41164)

The animation plays correctly the first time but after that the elements don't fade in a second time! Why is that?

`repeatRefresh` uses the end values of the animation as the starting values of the next iteration. In this case, the opacity of our text elements are all 0 at the end. So when the animation gets to the `.from()` the second time around, the opacity animates from a value of 0 to a value of 0 since the tween is relative.

What we want to do instead is always animate from a value of 0 to a value of 1 so here the easiest fix is to use a `.fromTo()`:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/LYGbwvR?default-tab=result\&theme-id=41164)

Now it does what we want. There are other solutions like using a `.set()` before the `.from()` but most often it's easiest to just use a `.fromTo()` in cases like this.

## Using fromTo() when from() or to() would work[​](#using-fromto-when-from-or-to-would-work "Direct link to Using fromTo() when from() or to() would work")

If you can, it 's better for performance, maintainability, and ease to use relative tweens like `.from()` or `.to()`. So don 't use `.fromTo()` unless you need to. `.fromTo()` tweens aren 't *bad*, but should only be used when needed.

## Not setting ALL transforms with GSAP[​](#transforms "Direct link to Not setting ALL transforms with GSAP")

If you are going to animate an element with GSAP, even the initial transform values (including on SVG elements) should be set with GSAP because it delivers better:

* **Accuracy** - The browser always reports computed values in *pixels*, thus it 's impossible for GSAP to discern when you use another unit like % or vw in your CSS rule. Also, computed values are in matrix() or matrix3d() which are inherently ambiguous when it comes to rotation and scale. The matrix for 0, 360, and 720 degrees are identical. A scaleX of -1 results in the same matrix as something with rotation of 180 degrees and scaleY of -1. There are infinite combinations that are identical, but when you set transform-related values with GSAP, everything is saved in a perfectly accurate way.
* **Performance** - GSAP caches transform-related values to make things super fast. Parsing all of the components from a computed value is more expensive.

If you are worried about a flash of unstyled content, you can handle that by using a technique that hides the element initially and then shows it via JavaScript as [this post covers](/resources/fouc.md). Or you can set the initial styles with CSS rules and *ALSO* set them in GSAP.

## Not using xPercent and yPercent[​](#not-using-xpercent-and-ypercent "Direct link to Not using xPercent and yPercent")

Did you know that you can *combine* percentage-based translation and other units? This is super useful if, for example, you 'd like to align the center of an element with a particular offset, like `{xPercent: -50, yPercent: -50, x: 100, y: 300}`. We often see people use percent values in the `x` and `y` properties which is technically possible but can cause confusion at times. For example, if you set `x` and `y` to `"-50%"` and then later you set `xPercent: -50`, you 'd see it move as if it 's at `xPercent: -100` because the `x` and `xPercent` both have `-50%`. Whenever you 're setting a percentage-based translation, it 's typically best to use the `xPercent` and `yPercent` properties.

```
// Not recommended
x: "50%",
y: "50%",

// Recommended
xPercent: 50,
yPercent: 50
```

## Recreating animations over and over[​](#recreating-animations-over-and-over "Direct link to Recreating animations over and over")

Creating your tweens and timelines beforehand has several advantages:

* **Performance** - Instead of having to create them right as they're needed, you can do it ahead of time. Additionally, you need fewer instances of animations. Most of the time you'd never notice, but it's good practice.
* **Simplified logic** - This is especially true when related to user interaction events.
* **Freedom** - Want to pause an animation when an event happens? Do it. Want to reverse an animation when the user does something? No problem. This sort of thing is much more difficult to handle when you create animations inside of event callbacks.

Most of the time when you create animations beforehand, you will want to keep them paused until they're needed. Then you can use control methods like `.play()`, `.pause()`, `.reverse()`, `.progress()`, `.seek()`, `.restart()`, and `.timeScale()` to affect their play state.

Here's a simple example:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/mdVPOQV?default-tab=result\&theme-id=41164)

For more information related to creating animations beforehand, you can see the [animating efficiently](https://css-tricks.com/tips-for-writing-animation-code-efficiently/) article.

One exception to this rule is when you need things to be dynamic, like if the initial values may vary. For example, if you're animating the height of the bars in a chart between various states and the user may click different buttons quickly, it'd make sense to create the animation each time to ensure they flow from whatever the current state is (even if it 's mid-tween) like the demo below.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ZEQBemg?default-tab=result\&theme-id=41164)

If you 're animating dynamically to a new position that 's updated very frequently, you might want to consider the [gsap.quickTo()](/docs/v3/GSAP/gsap.quickTo\(\).md) method.

## Adding tweens to completed timelines[​](#adding-tweens-to-completed-timelines "Direct link to Adding tweens to completed timelines")

A common pattern of mistakes that I've seen goes like this:

```
const tl = gsap.timeline()
tl.to(myElem, { x: 100 });

myElem.addEventListener("click", () => tl.to(myElem, { x: 300 }) );
```

Did you catch the mistake? If you add new tweens to a timeline that is already completed, they won't be called unless you re-run the timeline. Almost always in these situations you should just use control methods for a previously created animation or create a new animation instead (not using an existing timeline) following the guidelines that we covered in the previous section.

## Not using loops[​](#not-using-loops "Direct link to Not using loops")

If you want to apply the same effect to multiple elements (sections, cards, buttons, etc.) when a certain event happens to each one, you should almost always use a loop.

For example, don't use a selector like `"button"` when you want it to affect just one button.

For example, if you wanted to fire [an effect](/docs/v3/GSAP/gsap.effects.md) when each button is clicked:

```
// BAD: immediately animates ALL buttons at once!
gsap.effects.explode("button", { direction: "up", duration: 3 });

// GOOD: animation is specific to each button, and only when clicked
gsap.utils.toArray("button").forEach(btn => {
  btn.addEventListener("click", () => gsap.effects.explode(btn, { direction: "up", duration: 3 }))
});
```

Inside of this loop, you can use a selector that is scoped to the given element so that you 're only getting things INSIDE that element. For example:

```
gsap.utils.toArray(".container").forEach(container => {
  let info = container.querySelector(".information"),
      silhouette = container.querySelector(".silhouette .cover"),
      tl = gsap.timeline({ paused: true });
  
  tl.to(info, { yPercent: 0 })
    .to(silhouette, { opacity: 0 }, 0);
  
  container.addEventListener("mouseenter", () => tl.play() );
  container.addEventListener("mouseleave", () => tl.reverse() );
});
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/PoqRZOB?default-tab=result\&theme-id=41164)

## Importing GSAP incorrectly[​](#importing-gsap-incorrectly "Direct link to Importing GSAP incorrectly")

A common issue people face when using GSAP in a module environment is importing GSAP or its plugins incorrectly. Most of the time import errors error can be avoided by thoroughly reading the relevant parts of [the installation page](/docs/v3/Installation). I won 't copy all of the details into this post, but be sure to make use of that page if you 're facing any sort of import error. It even has a very handy GSAP install helper tool that can generate the correct import code to use in most environments.

## Using CSS transitions and GSAP on the same properties[​](#using-css-transitions-and-gsap-on-the-same-properties "Direct link to Using CSS transitions and GSAP on the same properties")

You should **definitely avoid** having CSS transitions applied to elements that you 're animating with GSAP. That 's terrible for performance because the browser would constantly be interrupting things. For example, let 's say you animate width to 500px from 100px. On every single tick (requestAnimationFrame), GSAP would set the interpolated value but the CSS transition would basically say *"NOPE! I won 't let you do that yet...I 'm gonna transition to that new value over the course of \_\_\_\_ seconds..."* and it 'd start interpolating. But on the very next tick, GSAP would set a new value and CSS transitions would interrupt and start over again, going to that new value. Over and over and over.

That would not only add a bunch of stress to the browser, but it 'd slow things down regarding the overall timing of the animation. For example, if the GSAP tween has a duration of 1 second and the CSS transition is also set to 1 second, that means it 'd stop moving after TWO seconds!

## Using old/verbose syntax[​](#using-oldverbose-syntax "Direct link to Using old/verbose syntax")

### Drop the Lite/Max[​](#drop-the-litemax "Direct link to Drop the Lite/Max")

I regularly see people using the old syntax even though they are loading [GSAP 3](https://gsap.com/blog/3-release-notes). Old habits die hard. Even though the old syntax still technically works, the new modern GSAP 3 syntax is sleeker and simpler. Plus the old syntax won 't be supported in GSAP 4 (which is ***far*** off in the future, but it 's still a good idea to write future-friendly code).

For example instead of using something that has Lite/Max in it, just use `gsap`:

```
// old
TweenLite.to()
TweenMax.from()
new TimelineMax()

// new
gsap.to()
gsap.from()
gsap.timeline()
```

### Use the string form for eases[​](#use-the-string-form-for-eases "Direct link to Use the string form for eases")

The shorter string form of eases requires less typing and lets you avoid extra import statements in module environments.

```
// old
Power2.easeOut
Sine.easeInOut

// new
"power2" // The default is .out
"sine.inOut"
```

### Duration belongs in the vars parameter[​](#duration-belongs-in-the-vars-parameter "Direct link to Duration belongs in the vars parameter")

Putting the duration inside of the vars parameter does require a bit more typing, but it makes things more readable and intuitive. GSAP's [defaults](/3-release-notes#defaults) and [effects](/docs/v3/GSAP/gsap.effects.md) are very helpful but you can't make use of them if you're putting the duration as the second parameter.

```
// old
gsap.to(elem, 1, { x: 100 });

// new
gsap.to(elem, { duration: 1, x: 100});

// using GSAP's defaults:
const tl = gsap.timeline({ defaults: { duration: 1 } });
tl.to(elem, { x: 100 }); // no duration necessary!
tl.to(elem, { y: 100, duration: 3 }); // easily overwrite the default value
```

For a more full listing of changes in GSAP 3, check out the [GSAP 3 Migration Guide](/resources/3-migration.md).

### Numerical values don't usually need to be strings[​](#numerical-values-dont-usually-need-to-be-strings "Direct link to Numerical values don't usually need to be strings")

For example if you want to set the x transform to 100 pixels, you don't need to say `x: "100px"`, you can just say `x: 100`. Simple!

The only time when you need to pass numerical values as strings are if you need to change the unit (like `x: "10vw"`) or pass in a complex value (like `transformOrigin: "0px 50px"`).

### The target of a tween can be a selector string[​](#the-target-of-a-tween-can-be-a-selector-string "Direct link to The target of a tween can be a selector string")

I often see people do something like this:

```
gsap.to(document.querySelectorAll(".box"), { x: 100 });
```

That will work but could be simplified by passing a selector string in as the target; GSAP will automatically use `.querySelectorAll()` to get a list of all of the elements that match. So the above can be written simple as

```
gsap.to(".box", { x: 100 });
```

You could also pass in a complex selector string like `".box, .card"` and it will select all boxes and cards. Or use an Array of elements so long as they are of the same type (selector string, variable reference, generic object, etc.).

## Conclusion[​](#conclusion "Direct link to Conclusion")

So how 'd you do? Is your GSAP code clear of these common mistakes? Hopefully you learned a few things. As always, if you need any help, the [GSAP Forums](https://gsap.com/community/) are a fantastic resource. We love to help people develop their animation superpowers.

If you 're looking for another great learning resource, read [how to animate efficiently](https://css-tricks.com/tips-for-writing-animation-code-efficiently/)!

Now go forth and tween responsibly!
