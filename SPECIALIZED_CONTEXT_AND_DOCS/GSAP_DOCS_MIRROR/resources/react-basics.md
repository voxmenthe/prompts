# Useful Patterns

Quick Start - useGSAP()

### useGSAP() hook

[view package on npm](https://www.npmjs.com/package/@gsap/react)

```
npm install @gsap/react
```

```
import gsap from "gsap";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(useGSAP);

const container = useRef();

useGSAP(() => {
  // gsap code here...
  gsap.to(".el", {rotation: 180}); // <-- automatically reverted

}, { scope: container }) // <-- scope for selector text (optional)
```

[Starter Templates](https://stackblitz.com/@GSAP-dev/collections/gsap-react-starters)

## useGSAP() is your best friend 💚[​](#usegsap-is-your-best-friend- "Direct link to useGSAP() is your best friend 💚")

In this guide we will be using our `useGSAP()` hook - the easiest way to use GSAP in React. It's a drop-in replacement for [React's Effects](https://react.dev/reference/react/useEffect) that automatically handles cleanup using [`gsap.context()`](https://gsap.com/docs/v3/GSAP/gsap.context\(\)/). [Cleanup is important](https://react.dev/learn/synchronizing-with-effects#triggering-animations) in React and this hook makes it simple.

```
import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(useGSAP);

const container = useRef();

useGSAP(() => {
  // gsap code here...
  gsap.to(".box", {x: 100}); // <-- automatically reverted

}, { scope: container }); // <-- easily add a scope for selector text (optional)
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/OJmQvLZ?default-tab=result\&theme-id=41164)

We cover [`useGSAP()`](https://www.npmjs.com/package/@gsap/react?activeTab=readme#gsapreact-for-using-gsap-in-react) in more detail in our [previous article](/resources/React.md#usegsap-hook-)

## Targeting elements with Refs[​](#targeting-elements-with-refs "Direct link to Targeting elements with Refs")

In order to animate, we need to tell GSAP which elements we want to target. The ***React way*** to access elements is by using **[Refs](https://reactjs.org/docs/refs-and-the-dom.html)**

Refs are a safe, reliable reference to a particular DOM node.

```
const boxRef = useRef();

useGSAP(() => {
 // Refs allow you to access DOM nodes
  console.log(boxRef) // { current: div.box }

  // then we can animate them like so...
  gsap.to(boxRef.current, {
    rotation: "+=360"
  });

});

return (
  <div className="App">
    <div className="box" ref={boxRef}>Hello</div>
  </div>
);
```

This can get messy

**However** - animation often involves targeting ***many*** DOM elements. If we wanted to stagger 10 different elements we'd have to create a Ref for **each** DOM node. This can quickly get repetitive.

```
// So many refs...
const container = useRef();
const box1 = useRef();
const box2 = useRef();
const box3 = useRef();

// ...just to do a simple stagger
useGSAP(() => {
  gsap.from([box1, box2, box3], {opacity: 0, stagger: 0.1});
});

return (
  <div ref={container}>
      <div ref={box1} className="box"></div>
      <div ref={box2} className="box"></div>
      <div ref={box3} className="box"></div>
  </div>
);
```

By defining a **scope** we can we leverage the ***flexibility*** of selector text with the **security** of Refs.

## Scoped Selectors[​](#scoped-selectors "Direct link to Scoped Selectors")

When we pass in a [Ref](https://react.dev/reference/react/useRef) as the scope, all selector text (like `".my-class"`) used in GSAP-related code inside the `useGSAP()` hook will be scoped accordingly, meaning your selector text will only select **descendants** of the container Ref. No need to create a Ref for every element!

```
// we only need one ref, the container. Use selector text for the rest (scoped to only find descendants of container).
const container = useRef();

useGSAP(() => {
    gsap.from(".box", {opacity: 0, stagger: 0.1});
}, { scope: container }) // <-- scope

return (
  <div ref={container}>
      <div className="box"></div>
      <div className="box"></div>
      <div className="box"></div>
  </div>
);
```

deep dive...

Refs or scoped selectors?

Targeting elements by using selector text like `".my-class"` in your GSAP-related code is much easier than creating a ref for each and every element that you want to animate - that's why we typically recommend using scoped selectors.

An exception would be if you're going to be **nesting** components and want to prevent against your selectors grabbing elements in child components.

In this example we've got two elements animating in the main App. A box targeted with a scoped class selector, and a circle targeted with a Ref. Both are animating just like we'd expect.

But - we've nested another component inside our app. This nested element **also** has child with a class name of '.box'. You can see that the nested box element is being targeted by the animation in the App's GSAP code, whereas the nested circle, which was targeted with a **Ref** isn't inheriting the animation.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/QWxBvMO?default-tab=result\&theme-id=41164)

## Reusing components[​](#reusing-components "Direct link to Reusing components")

Within a component based system, you may need more granular control over the elements you're targeting. You can pass props down to children to adjust class names or data attributes and target specific elements.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/RwVBWGW?default-tab=result\&theme-id=41164)

## Creating and controlling timelines[​](#creating-and-controlling-timelines "Direct link to Creating and controlling timelines")

Up until now we've just used refs to store references to DOM elements, but they're not just for elements. **Refs exist outside of the render loop** - so they can be used to store any value that you would like to persist for the life of a component.

In order to avoid creating a new timeline on every render, it's important to create the timeline inside the `useGSAP()` hook and store it in a `ref`.

```
function App() {
  const container = useRef();
  const tl = useRef();

  useGSAP(() => {
    tl.current = gsap
      .timeline()
      .to(".box", {
        rotate: 360
      })
      .to(".circle", {
        x: 100
      });
  }, { scope: container });

  return (
    <div className="app" ref={container}>
      <Box>Box</Box>
      <Circle>Circle</Circle>
    </div>
  );
}
```

Easy interactivity!

Storing the timeline in a Ref allows us to access the timeline in other [context-safe](/resources/React.md#making-your-animation-context-safe) functions or `useGSAP()` hooks to toggle the timeline direction.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/eYWGeGe?default-tab=result\&theme-id=41164)

## Controlling when React creates our animation.[​](#controlling-when-react-creates-our-animation "Direct link to Controlling when React creates our animation.")

`useGSAP()` uses an Effect internally, so we can control when the Effect should run by passing in an Array of dependencies. The default is an **empty Array**, which will run the effect after first render.

We can add props to our dependency array in order to control when our animation plays, this is useful if your animation needs to respond to a change in state. You can also force an animation to trigger on every render by passing in 'null' - but this is generally considered wasteful and should be avoided.

You can [read more about reactive dependencies](https://devtrium.com/posts/dependency-arrays) here.

```
// default uses [] internally and runs after first render
useGSAP(() => {
  gsap.to(".box-1", { rotation: "+=360" });
});

// runs after first render and every time `endX` changes
useGSAP(() => {
  gsap.to(".box-2", { x: endX  });
}, [endX]);

// this can be written out using the config object in order to pass in scope
useGSAP(() => {
  gsap.to(".box-2", { x: endX });
}, { dependencies: [endX], scope: container});

// 🔥 will run on every render
useGSAP(() => {
  gsap.to(".box-2", { rotation: "+=360"  });
}, null);
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/RwVZEMg?default-tab=result\&theme-id=41164)

## Reacting to changes in state[​](#reacting-to-changes-in-state "Direct link to Reacting to changes in state")

Now that we know how to control when an effect fires, we can use this pattern to respond to changes in our component. This is especially useful when passing down props.

```
function Box({ children, endX }) {
  const boxRef = useRef();

  useGSAP(() => {
    gsap.to(boxRef.current, {
      x: endX,
      duration: 3
    });
  }, [endX]);

  return (
    <div className="box" ref={boxRef}>
      {children}
    </div>
  );
}
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/poPrYGa?default-tab=result\&theme-id=41164)

info

If you define a dependency array and a dependency changes, the GSAP-related objects (animations, ScrollTriggers, etc.) **won’t** get reverted. They will only get reverted when the component is unmounted and the hook is torn down. If you'd prefer the context to be reverted every time the hook re-synchronizes (when any dependency changes), you can set `revertOnUpdate: true` in the config object.

You can see in the following demo that the reverted tween 'resets' back to it's original position in the DOM before each animation, whereas the un-reverted tween continues on from it's animated state.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/poGGORY?default-tab=result\&theme-id=41164)

## Performant interactive animations[​](#performant-interactive-animations "Direct link to Performant interactive animations")

If you're animating on an event like `mousemove` that fires a lot of times per second, we recommend using [`quickTo`](/docs/v3/GSAP/gsap.quickTo\(\).md) or [`quickSetter`](/docs/v3/GSAP/gsap.quickSetter\(\).md) for performance

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed//Rwvdowy/aa7e54bfc2a21a99c21230d884cb7176?default-tab=result\&theme-id=41164)
