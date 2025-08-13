# Call a function after viewport resizing stops (debounced)

"resize" events get dispatched **many** times while the user is dragging the browser window's edges, so if you run an expensive function on every resize event, it can really bog things down. Here's a function that will wait to call your function until a certain amount of time has elapsed since the user STOPPED resizing the window (by default, it's 0.2 seconds):

```
function callAfterResize(func, delay) {
  let dc = gsap.delayedCall(delay || 0.2, func).pause(),
    handler = () => dc.restart(true);
  window.addEventListener("resize", handler);
  return handler; // in case you want to window.removeEventListener() later
}
```

## Usage[​](#usage "Direct link to Usage")

```
callAfterResize(myFunction);
```
