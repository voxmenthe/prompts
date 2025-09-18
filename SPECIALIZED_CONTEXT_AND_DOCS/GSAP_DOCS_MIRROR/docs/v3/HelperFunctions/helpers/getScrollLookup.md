# Get the scroll position associated with an element (ScrollTrigger-aware)

This function even takes ScrollTrigger pinning into account in most situations. Feed it your target elements and it'll return a function that you can call later, passing it a specific one of those target elements and it'll return the scroll position. It even adjusts when the viewport resizes (responsive).

```
/*
Returns a FUNCTION that you can feed an element to get its scroll position.
- targets: selector text, element, or Array of elements
- config: an object with any of the following optional properties:
- start: defaults to "top top" but can be anything like "center center", "100px 80%", etc. Same format as "start" and "end" ScrollTrigger values.
- containerAnimation: the horizontal scrolling tween/timeline. Must have an ease of "none"/"linear".
- pinnedContainer: if you're pinning a container of the element(s), you must define it so that ScrollTrigger can make the proper accommodations.
*/
function getScrollLookup(
  targets,
  { start, pinnedContainer, containerAnimation }
) {
  let triggers = gsap.utils.toArray(targets).map((el) =>
      ScrollTrigger.create({
        trigger: el,
        start: start || "top top",
        pinnedContainer: pinnedContainer,
        refreshPriority: -10,
        containerAnimation: containerAnimation,
      })
    ),
    st = containerAnimation && containerAnimation.scrollTrigger;
  return (target) => {
    let t = gsap.utils.toArray(target)[0],
      i = triggers.length;
    while (i-- && triggers[i].trigger !== t) {}
    if (i < 0) {
      return console.warn("target not found", target);
    }
    return containerAnimation
      ? st.start +
          (triggers[i].start / containerAnimation.duration()) *
            (st.end - st.start)
      : triggers[i].start;
  };
}
```

## Usage[​](#usage "Direct link to Usage")

```
let getPosition = getScrollLookup(".section", {
  containerAnimation: horizontalTween,
  start: "center center",
});

// then later, use the function as many times as you want to look up any of the scroll position of any ".section" element
gsap.to(window, {
  scrollTo: getPosition("#your-element"),
  duration: 1,
});
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/popKYRW?default-tab=result\&theme-id=41164)
