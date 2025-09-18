# Step-by-step function calls progressively build timeline

Maybe you can't pre-build your entire timeline because you need to call individual functions in a sequenced fashion. Perhaps they each change the state of elements, creating an animation that must finish before the next step (function) is called. This helper function lets you organize your code quite easily into a simple sequence of arguments you pass, and you can even have a delay between each step:

```
function progressiveBuild() {
  let data = Array.from(arguments),
    i = 0,
    tl = gsap.timeline({
      onComplete: function () {
        let isNum = typeof data[i] === "number",
          delay = isNum ? data[i++] : 0,
          func = data[i++];
        typeof func === "function" && tl.add(func(), "+=" + delay);
      },
    });
  tl.vars.onComplete();
  return tl;
}
```

### Usage[​](#usage "Direct link to Usage")

```
progressiveBuild(
  step1,
  step2,
  1.5, // 1.5-second delay (sprinkle between any two functions)
  step3
);
```
