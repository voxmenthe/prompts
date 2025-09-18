# React

Why choose GSAP?

There are React-specific libraries that offer a more declarative approach. So why choose GSAP?

Animating imperatively gives you **a lot** more power, control and flexibility. Your imagination is the limit. You can reach for GSAP to animate everything from simple DOM transitions to SVG, three.js, canvas or WebGL.

Since GSAP is framework-agnostic, your animation superpowers transfer to ANY project; Vanilla JS, React, Vue, Angular, Webflow, whatever. You don't need to learn a React-specific library and then a completely different one for other projects. GSAP can be your trusted toolset wherever you go.

If you ever get stuck, our friendly [forum community](https://gsap.com/community//) is there to help. 💚

useGSAP() Walkthrough

Before we begin

* New to React? Check out [this tutorial](https://reactjs.org/tutorial/tutorial.html) from the React team.
* New to GSAP? Our [getting started guide](/resources/get-started.md) covers animation basics.
* In need of React help? Hit up the [Reactiflux community](https://www.reactiflux.com/) for expert advice.

## useGSAP() Hook 💚[​](#usegsap-hook- "Direct link to useGSAP() Hook 💚")

GSAP itself is completely framework-agnostic and can be used in any JS framework without any special wrappers or dependencies. However, this hook solves a few **React-specific** friction points for you so that you can just focus on the fun stuff. 🤘🏻

`useGSAP()` is a drop-in replacement for [`useEffect()`](https://react.dev/reference/react/useEffect) or [`useLayoutEffect()`](https://react.dev/reference/react/useLayoutEffect) that automatically handles cleanup using [`gsap.context()`](https://gsap.com/docs/v3/GSAP/gsap.context\(\)/). [Cleanup is important](https://react.dev/learn/synchronizing-with-effects#triggering-animations) in React and Context makes it simple.

Import the `useGSAP()` hook from `@gsap/react` and you're good to go! All GSAP animations, ScrollTriggers, Draggables, and SplitText instances [created when the useGSAP() hook executes](/resources/React.md#animating-on-interaction-) will be [reverted](https://gsap.com/docs/v3/GSAP/Timeline/revert\(\)) automatically when the component unmounts and the hook is torn down.

```
npm install @gsap/react
```

```
import { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP); // register the hook to avoid React version discrepancies 

const container = useRef();

useGSAP(() => {
	// gsap code here...
	gsap.to('.box', { x: 360 }); // <-- automatically reverted
},{ scope: container }); // <-- scope is for selector text (optional)
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/OJmQvLZ?default-tab=result\&theme-id=41164)

Deep dive

Why is cleanup so important?

Proper animation cleanup is very important with frameworks, but ***especially*** with React. React 18 runs in [strict mode](https://react.dev/reference/react/StrictMode) locally by default which causes your Effects to get called **TWICE**. This can lead to duplicate, conflicting animations or logic issues with [from](https://gsap.com/docs/v3/GSAP/gsap.from\(\)/) tweens if you don't revert things properly.

*The `useGSAP()` hook follows [React's best practices](https://react.dev/learn/synchronizing-with-effects#triggering-animations) for animation cleanup*

If you're interested in what's happening at a lower level, check out the [Context docs](https://gsap.com/docs/v3/GSAP/gsap.context\(\)/).

[View the package on npm](https://www.npmjs.com/package/@gsap/react)

SSR

This hook is safe to use in [Next](https://nextjs.org/) or other server-side rendering environments, provided it is used in client-side components. It implements the useIsomorphicLayoutEffect technique, preferring React's useLayoutEffect() but falling back to useEffect() if window isn't defined.

If you're using the app router / react server components, you need to drop a "use client" at the top of your file for useGSAP() to work

## Config Object[​](#config-object "Direct link to Config Object")

The second property is *optional*. You can pass either a [dependency array](https://devtrium.com/posts/dependency-arrays) - like [`useEffect()`](https://react.dev/reference/react/useEffect) - or a config object for more flexibility.

```
// config object offers maximum flexibility
useGSAP(() => {
		// gsap code here...
},{ dependencies: [endX], scope: container, revertOnUpdate: true });

useGSAP(() => {
	// gsap code here...
}, [endX]); // simple dependency array setup like useEffect, good for state-reactive animation

useGSAP(() => {
	// gsap code here...
}); // defaults to an empty dependency array '[]' and no scoping.
```

* ### Property

  ### Description

  #### dependencies[](#dependencies)

  Array / null : default \[]

  <br />

  The dependency array passed to the internal useEffect. [What's the dependency array for?](/resources/react-basics.md#controlling-when-react-creates-our-animation)

* #### scope[](#scope)

  [React ref](https://react.dev/reference/react/useRef) - Super useful!

  <br />

  Define a container as a scope in the config object to ensure that all GSAP selector text inside the the `useGSAP()` hook will be scoped to the descendants of that container. [Learn more...](/resources/react-basics.md#scoped-selectors)

* #### revertOnUpdate[](#revertOnUpdate)

  Boolean : default false

  <br />

  If you define a dependency array and a dependency changes, the GSAP-related objects (animations, ScrollTriggers, etc.) **won’t** get reverted. They will only get reverted when the component is unmounted and the hook is torn down. If you'd prefer the context to be reverted every time the hook re-synchronizes (when any dependency changes), you can set `revertOnUpdate: true`.

## Animating on interaction ✨[​](#animating-on-interaction- "Direct link to Animating on interaction ✨")

All GSAP animations, ScrollTriggers, Draggables, and SplitText instances that are created when the `useGSAP()` hook executes will automatically get added to the internal `gsap.context()` and reverted when the component unmounts and the hook is torn down. These animations are considered **'context-safe'**

**However**, if you create any animations that get called **after** the `useGSAP()` hook executes (like click event handlers, something in a `setTimeout()`, or anything delayed), those animations **will not** be context-safe.

DANGER! Animation added on click will not be cleaned up

Let's take a user triggered click event as an example: The animation inside the click event is only created when the user 'clicks'. Because this happens *after* the `useGSAP()` hook is executed (on mount) the animation won't get recorded, so it won't be included in the `context` for automatic cleanup.

```
const container = useRef();

useGSAP(() => {
	// ✅ safe, created during execution, selector text scoped
	gsap.to('.good', { x: 100 });
},{ scope: container });

// ❌ Unsafe! Created on interaction and not wrapped in contextSafe()
// animation will not be cleaned up
// Selector text also isn't scoped to the container.
const onClickBad = () => {
	gsap.to('.bad', { y: 100 });
};

return (
	<div ref={container}>
		<div className="good"></div>
		<button onClick={onClickBad} className="bad"></button>
	</div>
);
```

### Making your animation 'context-safe'[​](#making-your-animation-context-safe "Direct link to Making your animation 'context-safe'")

Let's tell `useGSAP()` about this animation so it can be added to the internal `gsap.context()`. Think of it like telling the Context when to hit the "record" button for any GSAP-related objects.

The `useGSAP()` hook exposes a couple of references for us:

* **context**: The `gsap.context()` instance that keeps track of all our animations.
* **contextSafe**: converts any function into a **context-safe** one so that any GSAP-related objects created while that function executes will be reverted when that Context gets reverted (cleanup). Selector text inside a context-safe function will also use the Context's scope. `contextSafe()` accepts a function and returns a new context-safe version of that function.

We can wrap up our click animation in the `contextSafe()` function in order to add it to the context. There are two ways to access this function:

#### 1) Using the returned object property (for outside useGSAP() hook)[​](#1-using-the-returned-object-property-for-outside-usegsap-hook "Direct link to 1) Using the returned object property (for outside useGSAP() hook)")

context-safe! Animation added on click event is added to the internal context

```
const container = useRef();

const { contextSafe } = useGSAP({ scope: container }); // we can pass in a config object as the 1st parameter to make scoping simple

// ✅ wrapped in contextSafe() - animation will be cleaned up correctly
// selector text is scoped properly to the container.
const onClickGood = contextSafe(() => {
	gsap.to('.good', { rotation: 180 });
});

return (
	<div ref={container}>
		<button onClick={onClickGood} className="good"></button>
	</div>
);
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ZEKJPLa?default-tab=result\&theme-id=41164)

#### 2) Using the 2nd argument (for inside useGSAP() hook)[​](#2-using-the-2nd-argument-for-inside-usegsap-hook "Direct link to 2) Using the 2nd argument (for inside useGSAP() hook)")

context-safe! Animation added on click event is added to the internal context

If you're manually adding event listeners, which is uncommon in React, don't forget to return a cleanup function where you remove your event listeners!

```
const container = useRef();
const badRef = useRef();
const goodRef = useRef();

useGSAP((context, contextSafe) => {
	// ✅ safe, created during execution
	gsap.to(goodRef.current, { x: 100 });

	// ❌ DANGER! This animation is created in an event handler that executes AFTER useGSAP() executes. It's not added to the context so it won't get cleaned up (reverted). The event listener isn't removed in cleanup function below either, so it persists between component renders (bad).
	badRef.current.addEventListener('click', () => {
		gsap.to(badRef.current, { y: 100 });
	});

	// ✅ safe, wrapped in contextSafe() function
	const onClickGood = contextSafe(() => {
		gsap.to(goodRef.current, { rotation: 180 });
	});

	goodRef.current.addEventListener('click', onClickGood);

	// 👍 we remove the event listener in the cleanup function below.
	return () => {
		// <-- cleanup
		goodRef.current.removeEventListener('click', onClickGood);
	};
},{ scope: container });

return (
	<div ref={container}>
		<button ref={badRef}></button>
		<button ref={goodRef}></button>
	</div>
);
```

## Starter Templates[​](#starter-templates "Direct link to Starter Templates")

Get started quickly by forking one of these starter templates:

* React

  ![](/img/react.svg)

  Simple CodePen starter

  [view ](https://codepen.io/GreenSock/pen/OJmQvLZ)

  <!-- -->

  [React](https://codepen.io/GreenSock/pen/OJmQvLZ)

* React

  ![](/img/react.svg)

  StackBlitz

  [view ](https://stackblitz.com/edit/gsap-react-basic-f48716?file=src%2FApp.js)

  <!-- -->

  [React](https://stackblitz.com/edit/gsap-react-basic-f48716?file=src%2FApp.js)

* React

  ![](/img/react.svg)

  StackBlitz + Plugins

  [view ](https://stackblitz.com/edit/react-iqmjfx?file=src%2FApp.js)

  <!-- -->

  [React](https://stackblitz.com/edit/react-iqmjfx?file=src%2FApp.js)

- [Stackblitz Collection](https://stackblitz.com/@GSAP-dev/collections/gsap-react-starters)

## Create a new React App[​](#create-a-new-react-app "Direct link to Create a new React App")

If you prefer to work locally, [Create React App](https://reactjs.org/docs/create-a-new-react-app.html) provides a comfortable setup for experimenting with React and GSAP.

1. To create a project, run:

   bash

   ```
   npx create-react-app gsap-app
   cd gsap-app
   npm start
   ```

2. Once the project is set up we can install GSAP and the special GSAP/React package through npm,

   bash

   ```
   # Install the GSAP library
   npm install gsap

   # Install the GSAP React package
   npm install @gsap/react

   # Start the project
   npm start
   ```

3. Then import it into our app.

   ```
   import { useRef } from 'react';

   import gsap from 'gsap'; // <-- import GSAP
   import { useGSAP } from '@gsap/react'; // <-- import the hook from our React package

   gsap.registerPlugin(useGSAP);

   export default function App() {
   	const container = useRef();

   	useGSAP(() => {
   		// gsap code here...
   		gsap.to('.box', { rotation: 180 }); // <-- automatically reverted
   	},{ scope: container }); // <-- scope for selector text (optional)

   	return (
   		<div ref={container} className="app">
   			<div className="box">Hello</div>
   		</div>
   	);
   }
   ```
