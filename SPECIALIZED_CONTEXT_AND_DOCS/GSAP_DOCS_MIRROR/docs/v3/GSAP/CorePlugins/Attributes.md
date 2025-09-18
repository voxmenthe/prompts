# Attributes

What are internal plugins?

GSAP animates attributes using an internal plugin, AttrPlugin is **automatically included in GSAP's core** and **doesn't have to be loaded using gsap.registerPlugin()**. You can think of internal plugins as just a part of GSAP.

### Description[​](#description "Direct link to Description")

GSAP allows you to easily tween any numeric attribute of a DOM element. For example, let's say your DOM element looks like this:

```
<rect id="rect" fill="none" x="0" y="0" width="500" height="400"></rect>
```

You could tween the "x", "y", "width", or "height" attributes like this:

```
gsap.to("#rect", {
  duration: 1,
  // x here refers to the x attribute
  attr: { x: 100, y: 50, width: 100, height: 100 },
  ease: "none",
  x: 200 // animate translateX() transform
});
```

You can tween an unlimited number of attributes simultaneously. Just use the associated property name inside the `attr:{}` object. GSAP will retain suffixes like "%" meaning you can tween values like `<rect width="50%"...>`.<br />**Caveat: you cannot do unit conversion with attributes (like px to %)**

Animating CSS

Do not attempt to animate CSS-related properties inside the attr object. GSAP handles [CSS](/docs/v3/GSAP/CorePlugins/CSS.md) differently internally. In the example above, the `x` outside the `attr:{}` object will animate the CSS transform, whereas the x inside the attr object will animate the underlying geometry - the x coordinate of the rect element.
