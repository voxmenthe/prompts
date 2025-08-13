# Weighted eases

Here's a useful function that'll let you feed in a ratio between -1 and 1 to any of the standard (non-configurable) eases to make them "weighted" in one direction or the other, as if it's pulling the ease curve toward the start or the end. A ratio of "0" won't be weighted either way, -1 would be weighted toward the "in" portion of the ease, and 1 is weighted toward the "out" portion.

```
function addWeightedEases() {
  let eases = gsap.parseEase(),
    createConfig = (ease) => (ratio) => {
      let y = 0.5 + ratio / 2;
      return (p) => ease(2 * (1 - p) * p * y + p * p);
    };
  for (let p in eases) {
    if (!eases[p].config) {
      eases[p].config = createConfig(eases[p]);
    }
  }
}

//example usage:
ease: "power2.inOut(0.5)"; // weighted halfway to the "out" portion
ease: "power2.inOut(-0.2)"; // weighted slightly to the "in" portion
ease: "power2.inOut(-1)"; // weighted ALL THE WAY to the "in" portion
ease: "power2.inOut(1)"; // weighted ALL THE WAY to the "out" portion
```

After you run that function once, you can basically configure any of the standard eases like `power2.inOut` or `power1.in` (or whatever) by adding parenthesis with a number indicating how you'd like to weight the ease in one direction or the other. And again, it works with any standard ease that doesn't already have a configuration option (like "steps()", "slow()", etc.)
