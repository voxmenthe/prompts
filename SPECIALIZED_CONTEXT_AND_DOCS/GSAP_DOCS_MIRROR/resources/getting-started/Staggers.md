# Staggers

staggering walkthrough

If you haven't tried creating staggered animations in GSAP yet, you're in for a treat - Staggers are totally configurable and **SUPER** powerful. If a tween has multiple targets, you can easily add some delay between the start of each animation:

## Simple Configuration[​](#simple-configuration "Direct link to Simple Configuration")

```
gsap.to('.box', {
	y: 100,
	stagger: 0.1 // 0.1 seconds between when each ".box" element starts animating
});
```

A value of `stagger: 0.1` would cause there to be 0.1 second between the start times of each tween. A negative value would do the same but backwards so that the last element begins first.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/LYdzaoz?default-tab=result\&theme-id=41164)

You can even stagger items that are laid out in a grid just by telling GSAP how many columns and rows your grid has!

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/rNVVGOa?default-tab=result\&theme-id=41164)

All tweens recognize a `stagger` property which can be a number, an object, or a function:

## Advanced configuration[​](#advanced-configuration "Direct link to Advanced configuration")

```
gsap.to('.box', {
	y: 100,
	stagger: {
		// wrap advanced options in an object
		each: 0.1,
		from: 'center',
		grid: 'auto',
		ease: 'power2.inOut',
		repeat: -1 // Repeats immediately, not waiting for the other staggered animations to finish
	}
});
```

To get more control, wrap things in a configuration object which can have any of the following properties (in addition to most of the [special properties](/docs/v3/GSAP/Tween.md#special-properties) that tweens have.

* ### Property

  ### Description

  #### amount[](#amount)

  \[Number]: The total amount of time (in seconds) that gets split among all the staggers. So if `amount` is `1` and there are 100 elements that stagger linearly, there would be 0.01 seconds between each sub-tween's start time. If you prefer to specify a certain amount of time between each tween, use the `each` property *instead*.

* #### each[](#each)

  \[Number]: The amount of time (in seconds) between each sub-tween's start time. So if `each` is `1` (regardless of how many elements there are), there would be 1 second between each sub-tween's start time. If you prefer to specify a **total** amount of time to split up among the staggers, use the `amount` property *instead*.

* #### from[](#from)

  \[String | Integer | Array]: The position in the Array from which the stagger will emanate. To begin with a particular element, for example, use the number representing that element's index in the target Array. So `from:4` begins staggering at the 5th element in the Array (because Arrays use zero-based indexes). The animation for each element will begin based on the element's proximity to the "from" value in the Array (the closer it is, the sooner it'll begin). You can also use the following string values: `"start"`, `"center"`, `"edges"`, `"random"`, or `"end"` ("random" was added in version 3.1.0). If you have a `grid` defined, you can specify decimal values indicating the progress on each axis, like `[0.5,0.5]` would be the center, `[1,0]` would be the top right corner, etc. Default: 0.

* #### grid[](#grid)

  \[Array | "auto"]: If the elements are being displayed in a grid visually, indicate how many rows and columns there are (like `grid:[9,15]`) so that GSAP can calculate proximities accordingly. Or use `grid:"auto"` to have GSAP automatically calculate the rows and columns using `element.getBoundingClientRect()` (great for responsive layouts). Grids are assumed to flow from top left to bottom right layout-wise (like text that wraps at the right edge). Or if your elements aren't arranged in a uniform grid, check out the [distributeByPosition() helper function](https://codepen.io/GreenSock/pen/gyWrPO?editors=0010) we created.

* #### axis[](#axis)

  \[string]: If you define a `grid`, staggers are based on each element's total distance to the "from" value on both the x and y axis, but you can focus on just one axis if you prefer (`"x"` or `"y"`). Use the demo above to see the effect (it makes more sense when you see it visually).

* #### ease[](#ease)

  \[String | Function]: The ease that distributes the start times of the animations. So `"power2"` would start out with bigger gaps and then get more tightly clustered toward the end. Default: `"none"`.

## Function[​](#function "Direct link to Function")

```
gsap.to('.box', {
	y: 100,
	stagger: function (index, target, list) {
		// your custom logic here. Return the delay from the start (not between each)
		return index * 0.1;
	}
});
```

Only use this if you need to run custom logic for distributing the staggers. The function gets called once for each target/element in the Array and should return the total delay from the starting position (not the amount of delay from the previous tween's start time). The function receives the following parameters:

1. **index** \[Integer] - The index value from the list.
2. **target** \[Object] - The target in the list at that index value.
3. **list** \[Array | NodeList] - The targets array (or NodeList).

## Repeat / Yoyo / Callbacks[​](#repeat--yoyo--callbacks "Direct link to Repeat / Yoyo / Callbacks")

Configuring Repeats

If you put your `repeat` in the top level of the vars object of your tween it will wait for **all** of the sub-tweens to finish before repeating the *WHOLE* sequence.

```
gsap.to(... {repeat:-1, stagger:{...})
```

If you prefer to have each sub-tween repeat independently (so that as soon as each one completes, it immediately repeats itself), simply nest the `repeat` (and `yoyo` if necessary) **inside** the advanced stagger object.

```
gsap.to(... {stagger:{repeat:-1, ...}});
```

The same thing is true for callbacks (e.g. onUpdate, onComplete, onStart) - including them **inside** of the stagger objects makes them fire **per element**. Think of it almost like the advanced stagger object is a vars object for the sub-tweens.

## FAQs[​](#faqs "Direct link to FAQs")

#### Are advanced staggers only useful for grids?

Absolutely not! You can get some slick effects by leveraging `ease` to add more organic spacing, or use the `from` property to have things emanate outward from a specific element index or the `"center"`. Even just using `amount` can be useful because it gives you tight control over when all the staggering will finish regardless of how many elements are in the target array.

#### Does `grid: "auto"` work for responsive layouts after resize?

No, stagger values get calculated immediately when the stagger method is called, so if the layout changes after that, you'd need to handle that logic on your own. For example, you could use a timeline's stagger method so that everything is in one instance, then have a resize listener that rewinds the timeline to its starting values, clears it, and then redoes the stagger. `timeline.time(0).clear(); timeline.to(...);` Ask in the [forums](https://gsap.com/community/) if you need any help with that.

#### What if my elements aren't in a uniform grid (gaps, different sizes, etc.) - can I stagger them based on position?

Sure, that's the perfect time to use a function-based value. In fact, we've created a [distributeByPosition() helper function](https://codepen.io/GreenSock/pen/gyWrPO?editors=0010) for this very case that should make it crazy simple for you!
