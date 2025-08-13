# Hook a Lottie animation up to ScrollTrigger

If you create an animation in After Effects and export it using [Lottie](https://airbnb.io/lottie/), you can hook it up to the scroll position with this handy function so that as the user scrolls, the animation progresses:

```
function LottieScrollTrigger(vars) {
  let playhead = { frame: 0 },
    target = gsap.utils.toArray(vars.target)[0],
    speeds = { slow: "+=2000", medium: "+=1000", fast: "+=500" },
    st = {
      trigger: target,
      pin: true,
      start: "top top",
      end: speeds[vars.speed] || "+=1000",
      scrub: 1,
    },
    ctx = gsap.context && gsap.context(),
    animation = lottie.loadAnimation({
      container: target,
      renderer: vars.renderer || "svg",
      loop: false,
      autoplay: false,
      path: vars.path,
      rendererSettings: vars.rendererSettings || {
        preserveAspectRatio: "xMidYMid slice",
      },
    });
  for (let p in vars) {
    // let users override the ScrollTrigger defaults
    st[p] = vars[p];
  }
  animation.addEventListener("DOMLoaded", function () {
    let createTween = function () {
      animation.frameTween = gsap.to(playhead, {
        frame: animation.totalFrames - 1,
        ease: "none",
        onUpdate: () => animation.goToAndStop(playhead.frame, true),
        scrollTrigger: st,
      });
      return () => animation.destroy && animation.destroy();
    };
    ctx && ctx.add ? ctx.add(createTween) : createTween();
    // in case there are any other ScrollTriggers on the page and the loading of this Lottie asset caused layout changes
    ScrollTrigger.sort();
    ScrollTrigger.refresh();
  });
  return animation;
}
```

### Usage[​](#usage "Direct link to Usage")

```
LottieScrollTrigger({
  target: "#animationWindow",
  path: "https://assets.codepen.io/35984/tapered_hello.json",
  speed: "medium",
  scrub: 2, // seconds it takes for the playhead to "catch up"
  // you can also add ANY ScrollTrigger values here too, like trigger, start, end, onEnter, onLeave, onUpdate, etc. See /docs/v3/Plugins/ScrollTrigger
});
```

**DEMO**

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/QWdjEbx?default-tab=result\&theme-id=41164)

Special thanks to Chris Gannon for his work on a [tool](https://github.com/chrisgannon/ScrollLottie) that inspired this.
