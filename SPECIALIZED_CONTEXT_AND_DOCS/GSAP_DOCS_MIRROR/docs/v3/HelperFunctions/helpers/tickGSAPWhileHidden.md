# Force GSAP to update while in hidden tab

Most browsers pause requestAnimationFrame() calls while a browser tab is hidden in order to reduce CPU/battery drain, but if you'd like GSAP to continue to update (at a reduced rate, typically about 2fps due to browsers throttling setInterval()/setTimeout()), you can use this function:

```
function tickGSAPWhileHidden(value) {
  if (value === false) {
    document.removeEventListener("visibilitychange", tickGSAPWhileHidden.fn);
    return clearInterval(tickGSAPWhileHidden.id);
  }
  const onChange = () => {
    clearInterval(tickGSAPWhileHidden.id);
    if (document.hidden) {
      gsap.ticker.lagSmoothing(0); // keep the time moving forward (don't adjust for lag)
      tickGSAPWhileHidden.id = setInterval(gsap.ticker.tick, 500);
    } else {
      gsap.ticker.lagSmoothing(500, 33); // restore lag smoothing
    }
  };
  document.addEventListener("visibilitychange", onChange);
  tickGSAPWhileHidden.fn = onChange;
  onChange(); // in case the document is currently hidden.
}

tickGSAPWhileHidden(true);
```
