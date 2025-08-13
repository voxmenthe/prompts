# Stop overscroll behavior, even on iOS Safari

It should be as simple as setting `overscroll-behavior: none` in your CSS, but iOS Safari won't cooperate! So here's a function that'll help stop the overscroll behavior:

```
function stopOverscroll(element) {
  element = gsap.utils.toArray(element)[0] || window;
  (element === document.body || element === document.documentElement) &&
    (element = window);
  let lastScroll = 0,
    lastTouch,
    forcing,
    forward = true,
    isRoot = element === window,
    scroller = isRoot ? document.scrollingElement : element,
    ua = window.navigator.userAgent + "",
    getMax = isRoot
      ? () => scroller.scrollHeight - window.innerHeight
      : () => scroller.scrollHeight - scroller.clientHeight,
    addListener = (type, func) =>
      element.addEventListener(type, func, { passive: false }),
    revert = () => {
      scroller.style.overflowY = "auto";
      forcing = false;
    },
    kill = () => {
      forcing = true;
      scroller.style.overflowY = "hidden";
      !forward && scroller.scrollTop < 1
        ? (scroller.scrollTop = 1)
        : (scroller.scrollTop = getMax() - 1);
      setTimeout(revert, 1);
    },
    handleTouch = (e) => {
      let evt = e.changedTouches ? e.changedTouches[0] : e,
        forward = evt.pageY <= lastTouch;
      if (
        ((!forward && scroller.scrollTop <= 1) ||
          (forward && scroller.scrollTop >= getMax() - 1)) &&
        e.type === "touchmove"
      ) {
        e.preventDefault();
      } else {
        lastTouch = evt.pageY;
      }
    },
    handleScroll = (e) => {
      if (!forcing) {
        let scrollTop = scroller.scrollTop;
        forward = scrollTop > lastScroll;
        if (
          (!forward && scrollTop < 1) ||
          (forward && scrollTop >= getMax() - 1)
        ) {
          e.preventDefault();
          kill();
        }
        lastScroll = scrollTop;
      }
    };
  if ("ontouchend" in document && !!ua.match(/Version\/[\d\.]+.*Safari/)) {
    addListener("scroll", handleScroll);
    addListener("touchstart", handleTouch);
    addListener("touchmove", handleTouch);
  }
  scroller.style.overscrollBehavior = "none";
}
```

## Usage[​](#usage "Direct link to Usage")

```
stopOverscroll();

// or specify an element
stopOverscroll("#viewport");
```
