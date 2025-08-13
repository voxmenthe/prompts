# Advanced techniques

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

Before we begin

Are you working with React and looking to ***really*** advance your GSAP animation skills? You're in the right place. This guide contains ***advanced*** techniques and some handy tips from expert animators in our community.

This is **not a tutorial**, so feel free to dip in and out as you learn. Think of it as a collection of recommended techniques and best practices to use in your projects.

If you're starting out we highly recommend reading our [foundational article](/resources/react-basics.md) first.

Why choose GSAP?

There are React-specific libraries that offer a simpler more declarative approach. So why choose GSAP?

Animating imperatively gives you **a lot** more power, control and flexibility. Your imagination is the limit. You can reach for GSAP to animate everything from simple DOM transitions to SVG, three.js, canvas or WebGL.

Since GSAP is framework-agnostic, your animation superpowers transfer to ANY project; Vanilla JS, React, Vue, Angular, Webflow, whatever. You don't need to learn a React-specific library and then a completely different one for other projects. GSAP can be your trusted toolset wherever you go.

Lastly, if you ever get stuck, our friendly forum community is there to help. 💚

***Going forward we will assume a comfortable understanding of both GSAP and React***.

## Component Communication[​](#component-communication "Direct link to Component Communication")

In the last article, we covered animating with the [useGSAP() hook](/resources/React.md#usegsap-hook-), and how to [create and control timelines](/resources/react-basics.md#creating-and-controlling-timelines) within a React component. But there are times where you may need to share a timeline across multiple components or construct animations from elements that exist in different components.

In order to achieve this, we need a way to communicate between our components.

**There are 2 basic approaches to this.**

1. a parent component can send down props, e.g. a timeline
2. a parent component can pass down a callback for the child to call, which could add animations to a timeline.

## Passing down a timeline prop[​](#passing-down-a-timeline-prop "Direct link to Passing down a timeline prop")

Note that we are using `useState `instead of `useRef` with the timeline. This is to ensure the timeline will be available when the child renders for the first time.

```
function Box({ children, timeline, index }) {
  const el = useRef();
  
  useGSAP(() => {
    // add 'left 100px' animation to timeline
    timeline && timeline.to(el.current, { 
      x: -100 
    }, index * 0.1);
    
  }, [timeline, index]);

  return <div className="box" ref={el}>{children}</div>;
}

function Circle({ children, timeline, index, rotation }) {
  const el = useRef();

  useGSAP(() => {
    // add 'right 100px, rotate 360deg' animation to timeline
    timeline && timeline.to(el.current, { 
      rotation: rotation, 
      x: 100 
    }, index * 0.1);
    
  }, [timeline, rotation, index]);

  return <div className="circle" ref={el}>{children}</div>;
}

function App() {
  const [tl, setTl] = useState();

  useGSAP(() => {
    const tl = gsap.timeline();
    setTl(tl);
  });

  return (
      <div className="app">
        <Box timeline={tl} index={0}>Box</Box>
        <Circle timeline={tl} rotation={360} index={1}>Circle</Circle>
      </div>
  );
}
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/XWReqpO?default-tab=result\&theme-id=41164)

## Passing down a callback to build a timeline[​](#passing-down-a-callback-to-build-a-timeline "Direct link to Passing down a callback to build a timeline")

```
function Box({ children, addAnimation, index }) {
  const el = useRef();

  useGSAP(() => {
    const animation = gsap.to(el.current, { x: -100 });
    addAnimation(animation, index);
  }, [addAnimation, index]);

  return (
    <div className="box" ref={el}>{children}</div>
  );
}

function Circle({ children, addAnimation, index, rotation }) {
  const el = useRef();

  useGSAP(() => {
    const animation = gsap.to(el.current, { rotate: rotation, x: 100 });
    addAnimation(animation, index);
  }, [addAnimation, index, rotation]);

  return (
    <div className="circle" ref={el}>{children}</div>
  );
}

function App() {
  const [reversed, setReversed] = useState(false);
  const [tl, setTl] = useState();

  useGSAP(() => {
    const tl = gsap.timeline();
    setTl(tl);
  }, []);

  const addAnimation = useCallback((animation, index) => {
    tl && tl.add(animation, index * 0.1);
  },[tl]);

  const toggleTimeline = contextSafe(() => {
    tl.current.reversed(!tl.current.reversed())
  });

  useGSAP(() => {
    // reverse the tim
    tl && tl.reversed(reversed);
  }, [reversed, tl]);

  return (
    <div className="app">
      <button onClick={() => setReversed(!reversed)}>Toggle</button>
      <Box addAnimation={addAnimation} index={0}>Box</Box>
      <Circle addAnimation={addAnimation} index={1} rotation="360">Circle</Circle>
    </div>
  );
}
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/poPWVpO?default-tab=result\&theme-id=41164)

## React Context[​](#react-context "Direct link to React Context")

Passing down props or callbacks might not be ideal for every situation.

The component you're trying to communicate with may be deeply nested inside other components, or in a completely different tree. For situations like this, you can use [React's Context.](https://reactjs.org/docs/context.html)

Whatever value your Context Provider provides will be available to any child component that uses the useContext hook.

Context

[React's Context](https://reactjs.org/docs/context.html) is not the same as [GSAP's Context](https://gsap.com/docs/v3/GSAP/gsap.context\(\)/)

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/wvdrERm?default-tab=result\&theme-id=41164)

## Creating reusable animations[​](#creating-reusable-animations "Direct link to Creating reusable animations")

Creating reusable animations is a great way to keep your code clean and scalable while reducing your app's file size.

### Declarative animation component[​](#declarative-animation-component "Direct link to Declarative animation component")

In this demo we're creating a component to handle the animation and then passing an object in to set the x value.

```
function FadeIn({ children, vars }) {
  const el = useRef();
  
  useGSAP(() => {
    animation.current = gsap.from(el.current.children, {
      opacity: 0,
      stagger,
      x
    });
  });
  
  return <span ref={el}>{children}</span>;
}
  
function App() {      
  return (
    <FadeIn vars={{ x: 100 }}>
      <div className="box">Box</div>
    </FadeIn>
  );
}
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/LYydbJV?default-tab=result\&theme-id=41164)

warning

If you want to use a React Fragment or animate a function component, you should pass in a ref for the target(s).

### RegisterEffect()[​](#registereffect "Direct link to RegisterEffect()")

GSAP also provides a way to create reusable animations with [`registerEffect()`](/docs/v3/GSAP/gsap.registerEffect\(\).md)

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/VwbXyON?default-tab=result\&theme-id=41164)

## Exit animations[​](#exit-animations "Direct link to Exit animations")

To animate elements that are exiting the DOM, we need to delay when React removes the element. We can do this by changing the component's state after the animation has completed.

```
function App() {      
  const app = useRef();
  const [active, setActive] = useState(true);

  const { contextSafe } = useGSAP({ scope: container });

  const remove = contextSafe(() => {
    gsap.to(".box", {
      opacity: 0,
      onComplete: () => setActive(false)
    });
  });
  
  return (
    <div className="app" ref={container}>
      <button onClick={ctx.remove}>Remove</button>
      { active ? <div ref={boxRef}>Box</div> : null }
    </div>
  );
}
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/QWvVzgB?default-tab=result\&theme-id=41164)

The same approach can be used when rendering elements from an array.

```
function App() {    
  
  const [items, setItems] = useState(() => [
    { id: 0, color: "blue" },
    { id: 1, color: "red" },
    { id: 2, color: "purple" }
  ]);

  const { contextSafe } = useGSAP();

  const remove = contextSafe((item, target) => {
    gsap.to(target, {
      opacity: 0,
      onComplete: () => removeItem(item)
    });
  });

  const removeItem = (value) => {
    setItems((prev) => prev.filter((item) => item !== value));
  };
  
  return (
    <div className="app" ref={container}>
      {items.map((item) => (
        <div key={item.id} onClick={(e) => ctx.remove(item, e.currentTarget)}>
          Click Me
        </div>
      ))}
    </div>
  );
}
```

Oh no! Layout shift...

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/LYyJqLO?default-tab=result\&theme-id=41164)

This works - but you may have noticed the layout shift - this is typical of exit animations. The [Flip plugin](/docs/v3/Plugins/Flip/.md) can be used to smooth this out.

## No more layout shifts with FLIP\![​](#no-more-layout-shifts-with-flip "Direct link to No more layout shifts with FLIP!")

In this demo, we're tapping into Flip's onEnter and onLeave to define our animations. To trigger onLeave, we have to set display: none on the elements we want to animate out.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/MWmqzjE?default-tab=result\&theme-id=41164)

## Custom Hooks[​](#custom-hooks "Direct link to Custom Hooks")

If you find yourself reusing the same logic over and over again, there's a good chance you can extract that logic into a [custom hook.](https://reactjs.org/docs/hooks-custom.html) Building your own Hooks lets you extract component logic into reusable functions. If you've made have any handy GSAP hooks - [let us know!](mailto:info@greensock.com?subject=React)

### useStateRef[​](#usestateref "Direct link to useStateRef")

This hook helps solve the problem of accessing stale values in your callbacks. It works exactly like useState, but returns a third value, a ref with the current state.

```
function useStateRef(defaultValue) {
  const [state, setState] = useState(defaultValue);
  const ref = useRef(state);

  const dispatch = useCallback((value) => {
    ref.current = typeof value === "function" ? value(ref.current) : value;
    setState(ref.current);
  }, []);

  return [state, dispatch, ref];
}
```

Usage:

```
const [count, setCount, countRef] = useStateRef(5);
const [gsapCount, setGsapCount] = useState(0);  

useGSAP(() => {
  gsap.to(".box", {
    x: 200,
    onRepeat: () => setGsapCount(countRef.current)
  });
}, {scope: app});
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/XWRqzbQ?default-tab=result\&theme-id=41164)

### useGSAP()[​](#usegsap "Direct link to useGSAP()")

and of course our own [`useGSAP()` hook!](/resources/React.md#usegsap-hook-)

Reach out!

If there is anything you'd like to see included in this article, or if you have any feedback or useful demos, please [let us know](mailto:info@greensock.com?subject=React) so that we can smooth out the learning curve for other animators.

Good luck with your React projects and happy tweening!
