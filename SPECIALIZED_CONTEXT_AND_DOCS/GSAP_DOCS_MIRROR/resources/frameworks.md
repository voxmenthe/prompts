# JS Frameworks

GSAP is a **framework-agnostic** animation library, that means that you can write the same GSAP code in [React](/resources/React.md), Vue, Angular or whichever framework you chose, the [core principles](/resources/get-started.md) won't change.

Using React? Handle cleanup automatically with our `useGSAP()` hook

Proper animation cleanup is important in most JS frameworks, but ***especially*** with React. React 18 runs in [strict mode](https://react.dev/reference/react/StrictMode) locally by default which causes your Effects to get called **TWICE**. This can lead to duplicate, conflicting animations or logic issues with [from](https://gsap.com/docs/v3/GSAP/gsap.from\(\)/) tweens if you don't revert things properly.

We created a hook that solves a few **React-specific** friction points for you so that you can just focus on the fun stuff.

*The `useGSAP()` hook follows [React's best practices](https://react.dev/learn/synchronizing-with-effects#triggering-animations) for animation cleanup*

[Learn More](/resources/React.md)

## Animation Cleanup[​](#animation-cleanup "Direct link to Animation Cleanup")

Whichever framework you use - it's always a good idea to **clean up your animations** by removing them when they're no longer needed. This way, you can make sure your animations play nicely and don't cause any hiccups like memory leaks or unexpected glitches.

Cleanup is of particular importance in Single Page Applications (SPAs) as they often involve dynamic content updates and renders - writing your animation code correctly will ensure your animations work reliably within your components lifecycle.

Demo framework - Vue We'll use Vue in these examples as the lifecycle events are nice and clear. :::

### A simple example - [`revert()`](/docs/v3/GSAP/Tween/revert\(\).md)[​](#a-simple-example---revert "Direct link to a-simple-example---revert")

You can call `revert()` on Tweens and Timeline's directly to kill the animation and return the targets to their pre-animation state. This includes removing any inline styles added by the animation.

Vue

```
let tween;

// create your animation when the component mounts
onMounted(() => {
	tween = gsap.to(el, { rotation: 360, repeat: -1 });
});

onUnmounted(() => {
	tween.revert(); // <- revert your animation when it unmounts
});
```

So far so good! **However**, this can get fiddly if you have a lot of animations.

### The most efficent way - [`gsap.context()`](/docs/v3/GSAP/gsap.context\(\).md)[​](#the-most-efficent-way---gsapcontext "Direct link to the-most-efficent-way---gsapcontext")

`gsap.context` makes cleanup nice and simple, all GSAP animations and ScrollTriggers created within the function get collected up so that you can easily `revert()` **ALL** of them at once.

Here's the structure:

Vue

```
let ctx;

onMounted(() => {
  // pop all your animatons inside a context
  ctx = gsap.context(() => {
    gsap.to(el,{rotation: 360, repeat: -1})

    let tl = gsap.timeline()

    tl.to(box,{x: 200})
    tl.to(box,{y: 500, duration: 2})
    tl.to(box,{rotation: 180, repeat: 2})
});

onUnmounted(() => {
  ctx.revert(); // <- Easy Cleanup!
});

// ...
```

You can use this pattern in any framework to make cleanup easier.

Take it for a spin - if you get stuck, pop over to the [forums](https://gsap.com/community/) and we'll give you a hand.

## Starter Templates[​](#starter-templates "Direct link to Starter Templates")

Looking for a jump start? Give one of these templates a go.

* React

  ![](/img/react.svg)

  React StackBlitz Collection

  [view ](https://stackblitz.com/@GSAP-dev/collections/gsap-react-starters)

  <!-- -->

  [React](https://stackblitz.com/@GSAP-dev/collections/gsap-react-starters)

* Next

  ![](/img/next.svg)

  NextJS StackBlitz Collection

  [view ](https://stackblitz.com/@GSAP-dev/collections/gsap-nextjs-starters)

  <!-- -->

  [Next](https://stackblitz.com/@GSAP-dev/collections/gsap-nextjs-starters)

* Vue

  ![](/img/vue.svg)

  Vue StackBlitz Collection

  [view ](https://stackblitz.com/@GSAP-dev/collections/gsap-vue3-starters)

  <!-- -->

  [Vue](https://stackblitz.com/@GSAP-dev/collections/gsap-vue3-starters)

* Nuxt 3

  ![](/img/nuxt.svg)

  Nuxt StackBlitz Collection

  [view ](https://stackblitz.com/@GSAP-dev/collections/gsap-nuxtjs-starters)

  <!-- -->

  [Nuxt 3](https://stackblitz.com/@GSAP-dev/collections/gsap-nuxtjs-starters)
