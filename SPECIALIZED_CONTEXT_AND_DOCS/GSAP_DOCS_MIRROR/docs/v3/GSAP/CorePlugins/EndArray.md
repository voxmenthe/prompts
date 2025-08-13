# EndArray

What are internal plugins?

EndArrayPlugin is an internal plugin, It is **automatically included in GSAP's core** and **doesn't have to be loaded using gsap.registerPlugin()**.

You can think of internal plugins as just a part of GSAP.

The endArray plugin enables you to tween an Array of numeric values to a different Array of numeric values with easing applied.

```
const arr = [1, 2, 3];

gsap.to(arr, {
  endArray: [5, 6, 7],
  onUpdate() {
    console.log(arr);
  },
});
```

If you have Arrays of uneven lengths, only the indexes that are in both Arrays will be animated.
