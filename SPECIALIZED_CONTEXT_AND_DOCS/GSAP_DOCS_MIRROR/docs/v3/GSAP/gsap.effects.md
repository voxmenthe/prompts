# gsap.effects

### Type : Object[​](#type--object "Direct link to Type : Object")

Once an effect has been [registered](/docs/v3/GSAP/gsap.registerEffect\(\).md), you can access it directly on the `gsap.effects` object like this:

```
//assumes that an effect named "explode" has already been registered

gsap.effects.explode(".box", {
  direction: "up", //can reference any properties that the author decides - in this case "direction"
  duration: 3,
});
```

Or, if you set `extendTimeline: true` on the effect when registering it, you'll even be able to call it DIRECTLY on timelines in order to have the results of the effect inserted into that timeline (see below). Effects make it easy for anyone to author custom animation code wrapped in a function (which accepts `targets` and a `config` object) and then associate it with a specific `name` so that it can be called anytime with new targets and configurations. For example, maybe we want to be able to make things fade (which is rather silly because it's so simplistic, but the goal here is to show how it could work):

```
// register the effect with GSAP:
gsap.registerEffect({
  name: "fade",
  effect: (targets, config) => {
    return gsap.to(targets, { duration: config.duration, opacity: 0 });
  },
  defaults: { duration: 2 }, //defaults get applied to any "config" object passed to the effect
  extendTimeline: true, //now you can call the effect directly on any GSAP timeline to have the result immediately inserted in the position you define (default is sequenced at the end)
});

// now we can use it like this:
gsap.effects.fade(".box");

// or directly on timelines:
let tl = gsap.timeline();
tl.fade(".box", { duration: 3 })
  .fade(".box2", { duration: 1 }, "+=2")
  .to(".box3", { x: 100 });
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/MWgmQmM?default-tab=result\&theme-id=41164)

GSAP provides 4 key services here:

* It parses the "targets" into an array. So if selector text is passed in, it becomes an array of elements passed to the effect function.
* It applies defaults to the config object every time. No need to add a bunch of if statements or apply the defaults yourself.
* It provides a centralized way of registering/accessing these "effects".
* If you set `extendTimeline: true`, the effect's name will be added as a method to GSAP's Timeline prototype, meaning that you can insert an instance of that effect directly into any timeline like:

```
//with extendTimeline: true
var tl = gsap.timeline();
tl.yourEffect(".class", { configProp: "value" }, "+=position");

//without extendTimeline: true, you'd have to do this to add an instance to the timeline:
tl.add(
  gsap.effects.yourEffect(".class", { configProp: "value" }),
  "+=position"
);
```

So it can save you a lot of typing if you're making heavy use of an effect in your sequences.

warning

**important**: any effect with `extendTimeline:true` **must** return a GSAP-compatible animation that could be inserted into a timeline (a Tween or Timeline instance)

Register Effects

For a quick overview of how to register effects, check out this video from the Snorkl.tv's [Creative Coding Club](https://www.creativecodingclub.com/bundles/creative-coding-club?ref=44f484) - one of the best ways to learn the basics of how to use GSAP.

Effects are also easily shared between different projects and people. To view effects that have already been created, check out [the CodePen collection](https://codepen.io/collection/bdffa09755cbd27a69b22771bd98e565/).

Here's an example of multiple pre-made fade effects being generated so that they can be reused later:

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/Rwajpyb?default-tab=result\&theme-id=41164)
