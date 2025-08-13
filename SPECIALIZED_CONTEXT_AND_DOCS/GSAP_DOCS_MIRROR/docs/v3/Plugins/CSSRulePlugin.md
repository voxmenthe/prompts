# CSSRule

warning

CSSRulePlugin has been **deprecated** in favor of using CSS variables which have [excellent browser support](https://caniuse.com/#feat=css-variables) these days. GSAP has native support for animating CSS variables, like:

```
gsap.to("html", { "--my-variable": 100, duration: 2 });
```

CSS variables demo:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/MoeLdj?default-tab=result\&theme-id=41164)

Be aware that animating a CSS variable, no matter the property targeted, will trigger a repaint, so use sparingly and be cautious of performance.

Allows GSAP to animate a raw style sheet rule which affects all objects of a particular selector rather than affecting an individual DOM element's inline styles.

For example, if you have a CSS class named `myClass` that sets `background-color` to `#FF0000`, you could tween that to a different color and *all* of the objects on the page that have a class of `myClass` would have their background color change. Typically it is best to use the regular CSSPlugin to animate CSS-related properties of individual elements so that you can get very precise control over each object, but sometimes it can be useful to animate the global rules themselves instead. For example, pseudo elements like `::after` and `::before` are impossible to reference directly in JavaScript, but you can animate them using CSSRulePlugin.

The plugin itself has a static `getRule()` method that you can use to grab a reference to the style sheet itself based on the CSS selector.

For example, let's say you have CSS like this:

```
.myClass {
    color: #FF0000;
}
.myClass::before {
    content: "This content is before.";
    color: #00FF00;
}
```

If you want to tween the color of the `.myClass:before` to blue. Make sure you load the CSSRulePlugin file and then do this:

```
var rule = CSSRulePlugin.getRule(".myClass::before"); //get the rule
gsap.to(rule, { duration: 3, cssRule: { color: "blue" } });
```

And if you want to get *all* of the `::before` pseudo elements, the `getRule()` will return an array of them, so I could do this:

```
gsap.to( CSSRulePlugin.getRule("::before"), {duration: 3, cssRule: {color: "blue"}});>
```

Keep in mind that it is typically best to tween a property that has already been defined in the specific rule that you're selecting because it cannot perform a calculated style (the combination of styles from other selectors that might pertain to similar elements). For example, if we didn't define any color initially for the `.myClass::before` and tried to tween its color to blue, it would start from `transparent` and go to `blue`. One way around this is to simply set your starting values explicitly in the tween by doing a `fromTo()`. That way there's no having to guess what the starting value should be when it isn't defined previously.

**Don't forget to wrap the values in a `cssRule: {}` object.**

Styles defined inside media queries may not be accessible or tweenable.

Alternatively, convert your pseudo-elements to real HTML elements and animate them directly like you would any other DOM elements.

## **Methods**[​](#methods "Direct link to methods")

|                                                                                                                        |                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| #### [CSSRulePlugin.getRule](/docs/v3/Plugins/CSSRulePlugin/methods/static-getRule\(\).md)( selector:String ) : Object | \[static] Provides a simple way to find the style sheet object associated with a particular selector like `".myClass"` or `"#myID"`. |
