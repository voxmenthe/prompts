# Modern React 19 Best Practices (with Vite)

Modern React (v19) introduces new hooks and patterns that simplify state management and side effects, letting us write more declarative and maintainable code. Using Vite as a fast build tool, we can quickly leverage these features in development. This report covers:

* **Alternatives to overusing useEffect** – new hooks like useEffectEvent, useActionState, useTransition, useDeferredValue, useSyncExternalStore, etc., with examples of when to use each.

* **Migration strategies for legacy code** – how to refactor function components that rely heavily on useEffect and how to convert class component lifecycles (e.g. componentDidMount/didUpdate/willUnmount) into idiomatic React 19 hooks.

* **Modern best practices in React 19** – including server components (when applicable), event handling patterns, state/derived state management, performance optimizations (Suspense, transitions, memoization), and component structure/co-location strategies.

* **Code examples and anti-patterns** – illustrating old vs. new approaches for each pattern.

Let’s dive into the details, using clear sections and examples for clarity.

## 1\. Modern Alternatives to useEffect

The useEffect Hook has long been the catch-all solution for side effects in React. We used it for everything from data fetching to subscriptions and DOM updates. However, this catch-all nature can lead to complex, hard-to-maintain code if overused[\[1\]](https://javascript.plainenglish.io/still-using-useeffect-check-out-react-19s-exciting-new-hooks-08a537100ed8?gi=8375feb7d77a#:~:text=Before%20we%20unpack%20the%20new,asynchronous%20tasks%20or%20managing%20forms). React 19 provides more specialized hooks and patterns that often eliminate the need for useEffect, making code clearer and avoiding pitfalls like stale state, redundant renders, and race conditions[\[2\]](https://javascript.plainenglish.io/still-using-useeffect-check-out-react-19s-exciting-new-hooks-08a537100ed8?gi=8375feb7d77a#:~:text=It%20has%20served%20well%20for,asynchronous%20tasks%20or%20managing%20forms)[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone). Below we outline modern alternatives and when to use them.

**Why look beyond useEffect?** Many effects in older code are not truly needed. Effects are meant for synchronizing with *external systems* (like APIs, browser APIs, or third-party libraries). If you’re just updating component state or responding to user input within React, you often *don’t need an Effect*[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone). Unnecessary effects can introduce bugs (stale closures, redundant state, double renders) and make code less efficient. Modern React encourages handling logic more declaratively or with dedicated hooks.

### 1.1 Use Event Handlers for User Actions (Instead of Effects)

One common anti-pattern is using useEffect to respond to something that a user just did. For example, you might set some state in an onClick handler, and then use an effect watching that state to perform an action (like sending an analytics event or triggering a follow-up state change). In most cases, it's clearer to handle the side effect directly inside the event handler itself. React docs note that you usually know what happened *in the event handler*, but by the time an Effect runs later, you’ve lost that context[\[4\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,in%20the%20corresponding%20event%20handlers). Handling user events in the event callback (onClick, onChange, etc.) avoids extra re-renders and keeps logic self-contained.

**Anti-pattern example:** Using an effect to handle a user event after state change.

function PurchaseButton() {  
  const \[purchased, setPurchased\] \= useState(false);  
  // ❌ Avoid: useEffect to respond to user action  
  useEffect(() \=\> {  
    if (purchased) {  
      // This runs after render, not immediately on click  
      sendPurchaseAnalytics();  
      alert("Purchase complete\!");  
    }  
  }, \[purchased\]);

  return \<button onClick={() \=\> setPurchased(true)}\>Buy\</button\>;  
}

In the above anti-pattern, clicking the button causes a state change, which then triggers an effect to perform side effects. This is unnecessary indirection.

**Preferred approach:** Do the work inside the event handler itself:

function PurchaseButton() {  
  const handleBuy \= () \=\> {  
    sendPurchaseAnalytics();  
    alert("Purchase complete\!");  
    // ...any state updates...  
  };  
  return \<button onClick={handleBuy}\>Buy\</button\>;  
}

Here we directly perform the side effect on click, which is simpler and avoids potential issues with stale state or extra renders[\[4\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,in%20the%20corresponding%20event%20handlers).

**When to use this pattern:** For most user interactions (form submissions, button clicks, etc.), handle the logic in the event handler. Only use an Effect if you truly need to synchronize with something external *whenever* a state/prop changes over time, independent of direct user input.

### 1.2 useEffectEvent – Stable Event Callbacks Inside Effects

Sometimes you *do* need to perform a side effect in response to changing state, but part of the effect logic shouldn’t re-run on every change. This is where **useEffectEvent** comes in. It lets you define a stable callback *inside* an effect for non-reactive logic. The callback always sees the latest values of props/state, but **does not** trigger re-running the effect when those values change[\[5\]](https://react.dev/learn/separating-events-from-effects#:~:text=Here%2C%20,of%20your%20props%20and%20state).

Think of useEffectEvent as a way to isolate event-like code within an effect. For example, suppose you open a WebSocket connection in an effect whenever roomId changes, and you want to show a notification when the connection succeeds using the current theme. Normally, you'd include theme as an effect dependency, which would cause the connection effect to restart whenever the theme changes – not ideal[\[6\]](https://react.dev/learn/separating-events-from-effects#:~:text=Show%20more). With useEffectEvent, you can wrap the notification logic so that it uses theme but doesn’t make the effect re-run:

import { useEffect, useEffectEvent } from 'react';

function ChatRoom({ roomId, theme }) {  
  // Define an Effect Event for the notification side-effect  
  const notifyConnected \= useEffectEvent(() \=\> {  
    showNotification('Connected\!', theme);  
  });

  useEffect(() \=\> {  
    const connection \= createConnection(roomId);  
    connection.on('connected', () \=\> {  
      // Call the stable callback when event occurs  
      notifyConnected();  
    });  
    connection.connect();  
    return () \=\> connection.disconnect();  
  }, \[roomId\]);  // Note: 'theme' is not a dependency here  
}

In this example, the connection effect depends only on roomId. The notifyConnected callback can use theme freely but won't cause re-subscription when theme changes[\[7\]](https://react.dev/learn/separating-events-from-effects#:~:text=return%20%28%29%20%3D). useEffectEvent ensures the notification always uses the latest theme without needing to restart the effect. It essentially gives you a **stable function** that doesn’t change identity and isn’t “reactive,” solving the stale closure problem in event callbacks[\[5\]](https://react.dev/learn/separating-events-from-effects#:~:text=Here%2C%20,of%20your%20props%20and%20state).

**When to use useEffectEvent:** Use it inside effects when you have some side-effect logic that depends on props/state that you **don’t** want to trigger the effect itself. It’s also useful as a replacement for patterns where you might otherwise use useCallback with no dependencies purely to get a stable function reference[\[8\]](https://github.com/facebook/react/issues/27793#:~:text=,constant%20reference%20to%20a). In React 19, useEffectEvent can handle many use cases that previously relied on useCallback, keeping your effect dependencies minimal and avoiding dependency lint warnings[\[9\]](https://react.dev/learn/separating-events-from-effects#:~:text=However%2C%20,a%20dependency%20of%20your%20Effect)[\[10\]](https://react.dev/learn/separating-events-from-effects#:~:text=Declaring%20an%20Effect%20Event).

### 1.3 useActionState – Managing Async Operations & Forms

React 19 introduces **useActionState** to simplify managing the lifecycle of asynchronous actions (such as form submissions, data mutations, etc.). In the past, handling an async operation often meant multiple useState hooks for loading flags, error messages, success flags, etc., plus an effect or a lot of boilerplate logic to reset these states at the right times[\[11\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=In%20earlier%20versions%20of%20React%2C,the%20status%20of%20each%20operation)[\[12\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=const%20handleSubmit%20%3D%20async%28%29%20%3D,setError%28%27Submission%20failed%27%29%3B). useActionState provides a concise way to track an action’s status (idle/loading/success/error) and perform the action.

**Old approach (pre-React 19):** multiple state variables and manual updates:

// Before: manually tracking an async action's state  
const \[isSubmitting, setIsSubmitting\] \= useState(false);  
const \[error, setError\] \= useState(null);  
const \[success, setSuccess\] \= useState(false);

async function handleSubmit() {  
  setIsSubmitting(true);  
  setError(null);  
  setSuccess(false);  
  try {  
    await submitApiCall();  
    setSuccess(true);  
  } catch (err) {  
    setError(err);  
  } finally {  
    setIsSubmitting(false);  
  }  
}

This approach works but is verbose and error-prone (easy to forget a state reset).

**Modern approach with useActionState:**

import { useActionState } from 'react';

function SubmitButton() {  
  const \[submitStatus, performSubmit\] \= useActionState(async () \=\> {  
    // This async function is the "action"  
    await submitApiCall();  
  });

  return (  
    \<button onClick={performSubmit} disabled={submitStatus.isLoading}\>  
      {submitStatus.isLoading ? 'Submitting...' : 'Submit'}  
      {submitStatus.error && \<p\>Error: {submitStatus.error.message}\</p\>}  
      {submitStatus.isSuccess && \<p\>Success\!\</p\>}  
    \</button\>  
  );  
}

Here, useActionState returns an object (submitStatus) containing flags like isLoading, isSuccess, and error, and also returns the trigger function (performSubmit)[\[13\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20SubmitButton%28%29%20)[\[14\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=With%20useActionState%3A). When you call performSubmit, it runs the async logic and automatically updates the status flags through the operation’s lifecycle. This eliminates the need for multiple useState calls and manual cleanup.

* **submitStatus.isLoading** is true while the action is in progress, and false otherwise.

* **submitStatus.isSuccess** indicates if the action completed successfully.

* **submitStatus.error** holds any error thrown.

Because useActionState encapsulates this logic, the component code is much cleaner[\[14\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=With%20useActionState%3A). You simply render UI based on the status flags. This hook shines for form submissions and any user-triggered async task. In fact, **React 19 “Actions”** (the concept of using async transitions for updates) use useActionState under the hood for common cases[\[15\]](https://react.dev/blog/2024/12/05/react-19#:~:text=Building%20on%20top%20of%20Actions%2C,cases%20for%20Actions%20in%20forms)[\[16\]](https://react.dev/blog/2024/12/05/react-19#:~:text=function%20ChangeName%28,), and you can use it directly to simplify form handling.

**Related form hooks:** React 19 also added useFormState and useFormStatus to improve form handling. useFormState can manage the entire state of a form’s inputs in one go (no more one useState per field)[\[17\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=Now%2C%20with%20the%20new%20,state%20when%20any%20field%20changes)[\[18\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20ProfileForm%28%29%20,name%3A%20%27%27%2C%20email%3A%20%27%27). useFormStatus lets any component inside a \<form\> check if the form is currently submitting (pending) or get the form data/result[\[19\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20LoginForm%28%29%20,%3D%20useFormStatus)[\[20\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=%3Cinput%20name%3D,button). Together with useActionState, these hooks make forms much more declarative:

* **useFormState** – replaces multiple field states by providing one form state object and an updater, keeping form input values in sync automatically[\[17\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=Now%2C%20with%20the%20new%20,state%20when%20any%20field%20changes)[\[21\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=const%20%5BformData%2C%20setFormData%5D%20%3D%20useFormState%28,name%3A%20%27%27%2C%20email%3A%20%27%27).

* **useFormStatus** – provides { pending, data, method, action } for the nearest form, so you can easily disable submit buttons or show loading spinners without prop-drilling loading state[\[19\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20LoginForm%28%29%20,%3D%20useFormStatus)[\[20\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=%3Cinput%20name%3D,button).

**When to use useActionState:** Use it for any asynchronous operation initiated by the user where you want to track loading and result. This includes form submissions (especially with \<form action={...}\> patterns in frameworks or your own form handler), saving data, etc. It’s essentially a built-in state machine for async tasks, replacing many effects that manage loading/error state manually. (If using a framework with built-in form actions, those might handle this for you; otherwise useActionState is great for SPAs built with Vite and React.)

### 1.4 useTransition – Marking Updates as Non-Blocking

React 18+ introduced **transitions** to differentiate urgent vs. non-urgent state updates. The **useTransition** hook lets you mark certain state updates as *transitions*, which tells React that these updates can be deferred so as not to block the UI responsiveness[\[22\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=We%20can%20use%20the%20useTransition,ensure%20a%20good%20user%20experience)[\[23\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=To%20make%20use%20of%20this%2C,inside%20the%20startTransition%20callback%20function). This is useful for expensive operations triggered by interactions – for example, filtering a large list in response to typing.

**Without transitions:** Every state update is treated as high priority. If you have a text input that triggers a heavy filter computation on each keystroke, the UI might lag because React synchronously re-renders the heavy list for each key.

**With useTransition:** You can update the input value immediately (urgent state), and update the filtered list in a transition (deferred state). This way, typing stays fast and React can pause or drop intermediate renders of the heavy list if new keystrokes come quickly[\[24\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=So%2C%20when%20I%20first%20enter,while%20typing%20into%20the%20textbox)[\[25\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=Now%2C%20if%20I%20enter%20another,while%20typing%20into%20the%20textbox).

**Example: Filtering a list with and without useTransition:**

function FilterableList({ items }) {  
  const \[query, setQuery\] \= useState('');  
  const \[filteredItems, setFilteredItems\] \= useState(items);

  const \[isPending, startTransition\] \= useTransition();

  function handleInput(e) {  
    const q \= e.target.value;  
    setQuery(q);                      // urgent update (input field value)  
    startTransition(() \=\> {  
      // deferred update (filter computation)  
      const result \= items.filter(item \=\> item.includes(q));  
      setFilteredItems(result);  
    });  
  }

  return (  
    \<\>  
      \<input value={query} onChange={handleInput} /\>  
      {isPending && \<p\>Updating list…\</p\>}  
      \<ItemList items={filteredItems} /\>  
    \</\>  
  );  
}

In this example, as the user types, the text field updates immediately, and the expensive filtering of a large list is done in a transition. React will keep isPending true while the transition is ongoing[\[26\]](https://react.dev/blog/2024/12/05/react-19#:~:text=const%20)[\[27\]](https://react.dev/blog/2024/12/05/react-19#:~:text=The%20async%20transition%20will%20immediately,while%20the%20data%20is%20changing). If the user types another character quickly, React can abandon the previous filtering work and start a new one – this prevents laggy typing[\[28\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=So%2C%20when%20I%20first%20enter,while%20typing%20into%20the%20textbox). The UI remains responsive, and you can show a loading state ("Updating list…") during the transition. This pattern avoids using an effect with a setTimeout or manual debounce to manage expensive updates; React handles it gracefully.

**When to use useTransition:** Use it when a state update triggers expensive re-renders or calculations that you want to defer. Common cases: filtering/searching through big lists, navigation between views (to keep the old UI while loading the new one), any large UI update resulting from a small interaction. In React 19, transitions are also used under the hood for the new “Actions” (async updates) to handle pending UI state automatically[\[29\]](https://react.dev/blog/2024/12/05/react-19#:~:text=function%20UpdateName%28)[\[30\]](https://react.dev/blog/2024/12/05/react-19#:~:text=The%20async%20transition%20will%20immediately,while%20the%20data%20is%20changing), but you can use useTransition directly in any SPA for performance tuning.

### 1.5 useDeferredValue – Deferring a Value Update

Related to transitions, **useDeferredValue** lets you take a value that may update frequently and defer its usage. It tells React *“keep this value updated, but if it changes very rapidly, allow a delay in updating the parts of the UI that use it.”* Essentially, it’s the hook form of a “debounce” using React’s concurrent rendering[\[31\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=useDeferredValue).

If you have a prop or state that triggers a heavy recalculation in a child component, but you can’t control how often it updates, you can pass a deferred copy of it. For example, if a parent component receives live updates (like live typing from a collaborative source) and passes a filter text to a heavy child component, the child can use useDeferredValue on that prop to avoid re-rendering on every tiny change.

**Example:**

function HeavyList({ searchTerm, items }) {  
  const deferredTerm \= useDeferredValue(searchTerm);  
  // This filtering will lag behind the actual searchTerm by a moment under heavy load  
  const filtered \= useMemo(  
    () \=\> items.filter(item \=\> item.includes(deferredTerm)),  
    \[items, deferredTerm\]  
  );  
  // ... render filtered list ...  
}

In this case, if searchTerm prop changes extremely fast, deferredTerm will update at a pace that allows React to maintain responsive rendering[\[32\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=This%20hook%20is%20very%20similar,will%20trigger%20a%20transition%20update)[\[33\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=Instead%2C%20we%20can%20use%20the,updates%20to%20take%20place%20faster). The UI may show results slightly behind the keystrokes, but the input can stay snappy. useDeferredValue is useful when **you receive a rapidly updating value and want to treat updates to it as low priority**. It’s like telling React “render this when you can”.

**When to use useDeferredValue:** Use it in a child component that gets a prop which can change rapidly and where updating the child is expensive. If you have control, useTransition (in the parent) is often the more direct way; but if not (say the parent is a third-party or complex), useDeferredValue in the child is a handy tool[\[34\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=a%20received%20prop%20value%20will,trigger%20a%20transition%20update). In practice, this hook is less common but good for fine-grained performance tweaks in concurrent React.

### 1.6 useSyncExternalStore – Subscribing to External Sources

Before React 18, to subscribe to an external data source (like a global store, a custom event emitter, or browser APIs), we would typically use useEffect to set up a subscription and update local state when the external source changed. This approach works but can be tricky to get right and isn't concurrent-safe. React 18+ provides **useSyncExternalStore** as a built-in way to subscribe to external data sources in a consistent, SSR-compatible way[\[35\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=return%20useSyncExternalStore).

**What it does:** useSyncExternalStore takes three arguments – a subscribe function, a snapshot getter for the client, and a snapshot getter for server – and returns the current value of the external store. React will handle subscribing, unsubscribing, and updating the component when the store changes, all with proper synchronization.

**Example – Online status (using browser events):**

// External subscribe function for browser online/offline events  
function subscribe(callback) {  
  window.addEventListener('online', callback);  
  window.addEventListener('offline', callback);  
  return () \=\> {  
    window.removeEventListener('online', callback);  
    window.removeEventListener('offline', callback);  
  };  
}

function useOnlineStatus() {  
  return useSyncExternalStore(  
    subscribe,  
    () \=\> navigator.onLine, // get current status on client  
    () \=\> true              // default for server render (assume online)  
  );  
}

function OnlineIndicator() {  
  const isOnline \= useOnlineStatus();  
  return \<div\>{isOnline ? "✅ Online" : "🚫 Offline"}\</div\>;  
}

In this example, useOnlineStatus custom hook uses useSyncExternalStore to subscribe to browser connectivity changes[\[36\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=function%20subscribe%28callback%29%20)[\[35\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=return%20useSyncExternalStore). The component OnlineIndicator then simply reads isOnline. This approach is **less error-prone** than manually using an effect to add event listeners and manage cleanup every time, and it integrates smoothly with React’s rendering (including server rendering)[\[37\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=).

**When to use useSyncExternalStore:** Use it whenever you need to hook React components up to an external subscription/source of truth: \- Reading from a global state store (Redux, Zustand, etc.) – many libraries now use useSyncExternalStore under the hood, or you can write your own adapter.  
\- Subscribing to browser APIs (e.g., online status, media queries, geolocation events, etc.).  
\- Integrating with other non-React event emitters or data streams.

It replaces the need for effects in these scenarios, providing a stable, thoroughly tested way to keep external data in sync with your component UI[\[37\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=). Just be sure to provide the snapshot getters correctly; for simple cases like above, it’s straightforward.

### 1.7 Other Notable Hooks and Patterns

* **useOptimistic:** This new hook in React 19 helps implement *optimistic UI updates*. Optimistic updates mean you update the UI immediately as if an action succeeded, and rollback if it fails. In the past, you’d manage this manually with state and maybe an effect. With useOptimistic, you can get an *optimistic state* and a setter that you call to apply a temporary update[\[38\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=As%20the%20name%20suggests%2C%20,the%20form%20is%20still%20submitting)[\[39\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=const%20,updateFn). If the action fails, React will revert to the actual state. It’s often used alongside Actions or useActionState to give instant feedback. **Use case:** e.g., you remove an item from a list UI immediately while the server request is still pending – the user sees it gone right away[\[40\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L190%20Best%20Practice%3A,instant%20feedback%20on%20their%20interactions). If the server errors, you bring it back. useOptimistic makes this pattern simpler by keeping track of the base state vs optimistic state.

* **Suspense and use():** Although not a hook per se, React 18/19 allow using Suspense for data fetching. The use(promise) utility can be called inside components to suspend until a promise resolves[\[41\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=import%20,boundary)[\[42\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=const%20DataComponent%20%3D%20%28,div%3E%3B). This is an alternative to an effect that fetches data and sets state. For example, inside a component you might do const data \= use(fetchData()) within a \<Suspense\> boundary. This way, you can show fallback UI while waiting, and you don’t need an effect or state at all for that data. When used in frameworks (Next.js) or with the upcoming React cache, this pattern drastically simplifies data loading. Best practice is to prefer use(promise) or framework data-fetching solutions over manual effect+state for loading data[\[43\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L244%20Best%20Practice%3A,cleaner%2C%20more%20readable%20async%20code)[\[44\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=) (more on this in Best Practices section).

* **useId:** Not an alternative to useEffect, but worth noting: useId helps generate stable unique IDs for accessibility or hydration. Use it instead of effects or random generators to ensure consistent IDs across server/client.

* **useLayoutEffect and useInsertionEffect:** These are variants of effects for specific needs (synchronous DOM mutations, style insertions). They are not “modern alternatives” but remain available for the few cases where you must measure DOM layout or inject styles before paint. Use them sparingly – in modern React you often can avoid useLayoutEffect by using CSS for layout or using refs to measure only when needed.

In summary, **React 19’s expanded hooks let you replace many useEffect uses with clearer, purpose-built solutions**. You might not need an effect to derive state, respond to user events, manage form submissions, coordinate external data, or handle performance concerns – there’s likely a hook or pattern that fits better. Next, we’ll see how to migrate older code to these modern patterns.

## 2\. Migration Guide for Legacy React Code

Updating a legacy React codebase to React 19 involves both syntactic updates (like using new hooks) and mindset shifts (moving from imperative patterns to declarative ones). Two common scenarios are:

1. **Functional components overusing useEffect:** Simplify or replace effects with modern patterns (as discussed above).

2. **Class components with lifecycle methods:** Convert them to function components with hooks in an idiomatic way.

Let’s tackle each.

### 2.1 From Effect-Heavy Function Components to Declarative Patterns

In older function components (React 16–18 era), one might use many useEffect hooks to handle various things: e.g., syncing props to state, triggering events on updates, fetching data, etc. When upgrading to React 19, **first evaluate whether each effect is truly necessary**[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone)[\[45\]](https://react.dev/reference/react/Component#:~:text=Note). The React docs literally say: *“If there is no external system involved, you shouldn’t need an Effect.”*[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone). Here’s a strategy to refactor such components:

* **Identify unnecessary state derivations:** If an effect is just computing a value from props or state and storing it in state (a derived state), remove that effect and state. Compute the value directly in render. For example, instead of using an effect to update fullName whenever firstName or lastName change, just compute const fullName \= firstName \+ " " \+ lastName in the component body (or useMemo if heavy). This removes redundant state and an entire effect[\[46\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,your%20props%20or%20state%20change)[\[47\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=function%20Form%28%29%20). It makes the code simpler and avoids an extra render for the effect update.

**Anti-pattern:**

// Before: using effect to derive combined state  
const \[fullName, setFullName\] \= useState("");  
useEffect(() \=\> {  
  setFullName(firstName \+ " " \+ lastName);  
}, \[firstName, lastName\]);  // unnecessary effect

**Refactor:**

// After: derive on the fly, no effect needed  
const fullName \= firstName \+ " " \+ lastName;

React will re-run the render and recompute fullName whenever dependencies change, which is sufficient. Removing such effects makes the code *“easier to follow, faster to run, and less error-prone.”*[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone)

* **Move event-driven logic out of effects:** As discussed, any effect that exists primarily to respond to a user action can likely be removed. E.g., if you had useEffect(() \=\> { if (modalOpen) focusInput(); }, \[modalOpen\]), you could instead call focusInput() at the moment you set modalOpen to true (perhaps via a ref or a callback). This way, you don’t rely on the effect timing. Evaluate each effect: if it’s effectively doing something that could be done in an event handler (onClick, onChange, etc.), refactor it to do so. This often simplifies state too (no need for flags just to trigger the effect)[\[4\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,in%20the%20corresponding%20event%20handlers).

* **Split up large effect logic:** If one useEffect is handling multiple concerns (e.g., one effect that on update does A, B, and C), consider splitting it into multiple effects or using appropriate hooks. For instance, an effect that both logs a message and fetches data on prop change could be two separate effects (or the logging moved to an event handler if it was user-initiated). React 19 encourages multiple smaller effects for unrelated logic rather than one monolithic effect with conditionals[\[48\]](https://react.dev/reference/react/Component#:~:text=This%20useEffect%20%20call%20is,example%20you%20can%20play%20with). This makes dependencies clearer and avoids running unrelated code when only one thing changes.

* **Use new hooks where applicable:** Go through each effect’s purpose and see if a React 19 hook can replace it:

* Effect doing an async API call? Could use Suspense/use() or at least encapsulate in a custom hook (useData) to manage loading and avoid race conditions (or use an external data library).

* Effect setting up a subscription? Use useSyncExternalStore or a custom hook wrapping it (as shown with useOnlineStatus).

* Effect handling form submission state? Use useActionState or useFormStatus.

* Effect scheduling a delayed action? Use useTransition or useDeferredValue for concurrency-friendly scheduling.

* Effect primarily just calling a function (like logging) when something changes? Possibly use useEffectEvent to isolate that or simply reconsider if it's needed at all.

**Example Refactor:** Suppose we have a search component that uses an effect to fetch results whenever the query changes, and also uses state to track loading and errors:

function SearchPage({ query }) {  
  const \[results, setResults\] \= useState(\[\]);  
  const \[loading, setLoading\] \= useState(false);  
  const \[error, setError\] \= useState(null);

  useEffect(() \=\> {  
    let ignore \= false;  
    setLoading(true);  
    setError(null);  
    fetch(\`/api/search?query=${query}\`)  
      .then(res \=\> res.json())  
      .then(data \=\> { if (\!ignore) setResults(data); })  
      .catch(err \=\> { if (\!ignore) setError(err); })  
      .finally(() \=\> { if (\!ignore) setLoading(false); });  
    return () \=\> { ignore \= true; };  
  }, \[query\]);

  // ... render results or error ...  
}

This is a fairly standard pattern pre-React 18\. To modernize it: \- We recognize this is data fetching. In a React 19 context, if using a framework or willing to use Suspense, we could do away with the effect entirely. For example, using a custom hook or use(). \- We might use a custom hook useData(url) that wraps the fetching logic with an effect internally (to encapsulate the race condition handling). Or use a library like React Query (though focusing on core React here). \- If we had server components available (e.g., Next.js 13+ with React 19), we might fetch on the server and pass results in as props, avoiding client-side effect for initial load.

A middle-ground refactor using transitions could be:

function SearchPage({ query }) {  
  const \[startTransition, isPending\] \= useTransition();  
  const \[results, setResults\] \= useState(\[\]);  
  const \[error, setError\] \= useState(null);

  // When query changes, start a transition to fetch new data  
  useEffect(() \=\> {  
    setError(null);  
    startTransition(() \=\> {  
      fetch(\`/api/search?query=${query}\`)  
        .then(r \=\> r.json())  
        .then(data \=\> setResults(data))  
        .catch(err \=\> setError(err));  
    });  
  }, \[query, startTransition\]);

  // render: maybe show spinner if isPending, etc.  
}

This uses useTransition to mark the fetch as a low-priority update (so if the user types another character in the search box, we won’t block the UI). We still use an effect here, but we removed the explicit loading state (isPending from the transition tells us if it’s loading) and let React handle some concurrency. We also removed the manual cleanup with ignore by relying on the transition behavior (if a new query comes, the old transition can be canceled). For a fully modern approach, we could use Suspense and a data cache or use(), which would eliminate the effect entirely and let React handle the timing.

**General tip:** After refactoring, run ESLint (with the React Hooks rules) to catch any missing dependencies or issues. React 19’s stricter linting and guidance (and even an upcoming React compiler) will help ensure you aren’t introducing new bugs when removing effects. The goal is cleaner code: one that *“avoids bugs caused by different state variables getting out of sync”*[\[49\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=code%29%2C%20and%20less%20error,what%20should%20go%20into%20state) and doesn’t re-render unnecessarily.

### 2.2 Converting Class Lifecycles to Hooks (React 19 Edition)

If you still have class components, migrating them to function components with hooks will unlock all the modern patterns. React 19 still supports class components, but the recommendation is to use function components for new code[\[50\]](https://react.dev/reference/react/Component#:~:text=,using%20them%20in%20new%20code). Here’s how to map common lifecycle methods to hooks and patterns:

* **componentDidMount:** This runs once after the first render. The equivalent in a function component is a useEffect with an empty dependency array (useEffect(() \=\> { ... }, \[\])). Place any setup code here (data fetching, subscribing to events, etc.) that should happen on mount. For example, a class using componentDidMount to fetch data would become a useEffect in the functional version. If the effect returns a cleanup function, that covers componentWillUnmount (see below).

* **componentDidUpdate:** This runs after *every* update (except the first) and receives prevProps/prevState. In hooks, you achieve the same by using useEffect with specific dependencies. If you want code to run when a certain prop or state changes, include it in the dependency array. You typically don’t need the equivalent of prevProps because you can compare values within the effect if necessary, or use multiple effects for different concerns. For example, a class might do:

* componentDidUpdate(prevProps) {  
    if (this.props.value \!== prevProps.value) { /\* do something \*/ }  
  }

* In a hook, you could write:

* useEffect(() \=\> {  
    // do something when value changes  
  }, \[value\]);

* The effect runs only when value changes (React compares dependencies by value). If you had more complex logic depending on multiple props, you can include all as dependencies – the effect will run when any of them change. If the class’s componentDidUpdate had a lot of conditional branches for various props, consider splitting into multiple useEffects, each one focused on a specific prop or concern[\[48\]](https://react.dev/reference/react/Component#:~:text=This%20useEffect%20%20call%20is,example%20you%20can%20play%20with).

* **componentWillUnmount:** This is for cleanup (removing event listeners, canceling timers, etc.). In a hook, any cleanup is returned from the same useEffect that did the setup. For example, a class that sets up a subscription in didMount and removes it in willUnmount becomes:

* useEffect(() \=\> {  
    const subscription \= subscribeToSomething();  
    return () \=\> {  
      subscription.unsubscribe(); // cleanup on unmount  
    };  
  }, \[\]);  // empty deps \= run on mount, cleanup on unmount

* If your class had multiple unmount cleanup tasks from different lifecycles, ensure each corresponding effect returns a cleanup.

* **componentDidMount \+ componentDidUpdate combined:** Often classes use componentDidUpdate to handle changes to the same things set up in didMount. For example, a class might connect to a server in didMount, and if a roomId prop changes, reconnect in didUpdate, plus disconnect in willUnmount. In hooks, you can unify that logic in one useEffect that depends on roomId (and any other relevant values). The effect’s cleanup will handle both unmount and before next re-run. The React docs demonstrate this: to migrate a class that connects to a chat server in didMount/didUpdate and disconnects in willUnmount, you use one useEffect that includes \[roomId\] in dependencies[\[51\]](https://react.dev/reference/react/Component#:~:text=In%20the%20above%20example%2C%20the,logic%20as%20a%20single%20Effect)[\[52\]](https://react.dev/reference/react/Component#:~:text=useEffect%28%28%29%20%3D). That effect will run on mount and whenever roomId changes, and its cleanup will run on unmount or before a re-run (disconnecting the previous connection) – covering the class logic exactly[\[52\]](https://react.dev/reference/react/Component#:~:text=useEffect%28%28%29%20%3D)[\[53\]](https://react.dev/reference/react/Component#:~:text=%2F%2F%20).

For example, class:

componentDidMount() { connect(roomId); }  
componentDidUpdate(prevProps) {  
  if (prevProps.roomId \!== this.props.roomId) {  
    disconnect(prevProps.roomId);  
    connect(this.props.roomId);  
  }  
}  
componentWillUnmount() { disconnect(this.props.roomId); }

becomes:

useEffect(() \=\> {  
  connect(roomId);  
  return () \=\> disconnect(roomId);  
}, \[roomId\]);

React takes care of calling the cleanup before the next effect if roomId changes, so the old connection is closed before opening a new one.

* **shouldComponentUpdate:** In classes, this method can be used for performance to skip re-renders. With function components, you don’t have this method; instead, you can wrap the component in React.memo to achieve a similar effect (skipping re-render if props didn’t change). Hooks like useMemo and useCallback also help optimize expensive calculations or stabilize function props passed down. However, a big shift in React 18+ is automatic batching and other performance improvements; you should only optimize if you have a proven performance problem. (Also, React 19’s upcoming compiler might handle many cases of memoization automatically[\[54\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20Write%20straightforward%20JavaScript,compiler%20might%20need%20a%20hint).)

* **getDerivedStateFromProps:** This static lifecycle is used to derive state from props before render. In modern React, you often don't need this; instead, compute what you need during render or use useEffect if side effects are needed. If you truly need to adjust state based on prop changes (e.g., reset a form if a prop ID changes), you can use an effect that checks the prop and calls setState. But more commonly, you manage such logic differently (key the component by prop so it remounts, or explicitly call a function on prop change). There isn’t a direct hook equivalent, because deriving state is usually handled more cleanly via props or context.

* **componentDidCatch / error boundaries:** There is no hook equivalent for error boundaries yet – you still need class components for error boundary components (as of React 19, though there’s talk of a future hook). Plan to leave error boundary components as classes or use a library for error boundaries if you want functional ones.

**Migration workflow:**

1. **Refactor state:** Convert this.state to one or more useState or useReducer hooks. Each independent piece of state becomes its own hook. For example, this.state \= {count: 0, text: ""} in a class might become two useState calls in the function: const \[count, setCount\] \= useState(0); const \[text, setText\] \= useState("");. If state pieces are interrelated (updates to one often come with updates to another), consider using useReducer to manage them together.

2. **Refactor lifecycle methods:** as discussed, map them to effects:

3. Combine mount/unmount logic in one useEffect.

4. If componentDidUpdate was doing something on prop change, use an effect with that prop as dependency.

5. If multiple lifecycles were used for different things (e.g., one for subscribing to a store, another for logging analytics), you can use multiple effects in the function component, one for each concern (this is often cleaner).

6. **Props and context:** In a class, you access props via this.props and context via this.context or Context.Consumer. In a function, just use the props directly as function arguments, and use useContext(MyContext) to get context values (this replaces static contextType and so on).

**Example:** Migrating a simple class:

// Class component  
class Welcome extends React.Component {  
  componentDidMount() {  
    console.log("Mounted\!");  
  }  
  componentWillUnmount() {  
    console.log("Will unmount.");  
  }  
  render() {  
    return \<h1\>Hello, {this.props.name}\</h1\>;  
  }  
}

**Functional equivalent:**

function Welcome({ name }) {  
  useEffect(() \=\> {  
    console.log("Mounted\!");  
    return () \=\> {  
      console.log("Will unmount.");  
    };  
  }, \[\]); // empty deps \-\> run on mount/unmount only

  return \<h1\>Hello, {name}\</h1\>;  
}

This covers the same lifecycle events in a hook. If the class did more (e.g., responded to prop changes), we’d add dependencies or additional effects accordingly.

The official React docs offer step-by-step examples of migrating classes with state and lifecycles to hooks[\[55\]](https://react.dev/reference/react/Component#:~:text=Pitfall)[\[51\]](https://react.dev/reference/react/Component#:~:text=In%20the%20above%20example%2C%20the,logic%20as%20a%20single%20Effect). Key points from those guides: \- *Verify* what your componentWillUnmount does relative to componentDidMount – ensure the effect’s cleanup does the same (release resources, remove listeners)[\[56\]](https://react.dev/reference/react/Component#:~:text=First%2C%20verify%20that%20your%20componentWillUnmount,is%20missing%2C%20add%20it%20first). \- *Verify* what triggers your componentDidUpdate – include all those things as dependencies in the effect, or use multiple effects if easier to separate concerns[\[57\]](https://react.dev/reference/react/Component#:~:text=Next%2C%20verify%20that%20your%20componentDidUpdate,and%20state%2C%20fix%20that%20first). \- After converting, test that the behavior is identical.

**Idiomatic React 19 hooks:** Once classes are converted, you can further modernize by applying the new hooks. For instance, if your class was doing some manual async handling, now you can integrate useActionState or transitions as appropriate, rather than a raw useEffect. But even a straight one-to-one conversion (class lifecycles \-\> basic useEffect) is a good first step. From there, leverage the earlier section’s advice to refine the use of effects (e.g., maybe you realize “Oh, I don’t actually need an effect here at all now\!”).

Finally, run your app’s tests or do manual testing after migration. Ensure that for each lifecycle path the class had, your function component \+ hooks cover it. Once migrated, you’ll likely notice the function version is shorter and more focused. With practice, you’ll “think in effects” differently than lifecycle methods, which often leads to simpler logic overall.

## 3\. Modern Best Practices for React 19 (with Vite)

Now that we have the tools and migration strategies, let's outline current best practices for building React 19 applications. These cover architectural considerations, state management, performance, and project structure. Even if you’re not using a framework like Next.js, it’s important to be aware of trends like Server Components because they influence how we write React apps going forward.

### 3.1 Leverage Server Components (When Available)

React 19 was designed with **React Server Components (RSC)** in mind, although RSC requires a framework (e.g. Next.js, Remix) or custom setup to use. In traditional SPA setups (e.g. React \+ Vite without Next), you won't directly use server components, but the concept still guides best practices. The idea is: **move as much work to the server as possible**, delivering a minimal, hydration-efficient payload to the client.

If you *are* using a framework that supports RSC, prefer to fetch data and assemble the UI tree on the server for the initial load. Server Components have zero client-side bundle cost and can directly access databases or files, which means you can skip many client-side data fetching effects altogether[\[58\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=While%20not%20part%20of%20the,can%20access%20backend%20resources%20directly)[\[59\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20For%20static%20content,only%20APIs). For example, in Next.js 13+, your page can be an async Server Component that fetches data (via await in the component) and renders the result – no loading spinners or effect needed for that initial load. Your client components can then be smaller and focused purely on interactivity.

**Best Practice:** “For static content and initial data fetching, default to Server Components. Use Client Components ("use client") only when you need interactivity, state, or browser-only APIs.”[\[59\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20For%20static%20content,only%20APIs) In other words, if a component doesn't need to handle user input or state, it might be a good candidate to be a server-rendered piece of UI.

In a Vite context (which by itself doesn't provide RSC), you can still apply the spirit of this: \- Do heavy data fetching on the server side (for example, via an SSR process or at build time) whenever you can, to send HTML to the client that’s already populated. \- If full SSR isn’t feasible, consider using libraries like React Query or SWR on the client which at least handle caching, deduping, and prefetching elegantly, so you don’t write low-level effect logic for data. \- Use suspense boundaries for data fetching (with use() or external libs) to improve user experience with built-in loading states, rather than imperative loading spinners in every component.

**Example (Next.js Server Component)** – Even though you might not use Next in a Vite project, it's illustrative:

// Server Component (e.g., Next.js app/page.js)  
export default async function ProductsPage() {  
  const products \= await db.query("SELECT \* FROM products");  
  return (  
    \<main\>  
      \<h1\>Our Products\</h1\>  
      {/\* ProductList could be a client component for interactivity,   
          but the data is already fetched here \*/}  
      \<ProductList products={products.rows} /\>  
    \</main\>  
  );  
}

This code runs on the server – by the time it reaches the browser, the HTML already includes the products. No client useEffect needed to load them. If ProductList is a client component, it simply receives the list via props.

For a Vite SPA, you might simulate this by server-rendering the initial HTML with data (using something like ReactDOMServer) or by embedding initial data in the page. If that’s not possible, the next best thing is to fetch as early as possible and use Suspense to manage the loading UI.

The bottom line: **be “Server Component-aware.”** Structure your app so that pure data-rendering components can be easily moved to the server or be used with Suspense. This might mean separating a container (that fetches data) from a presentational component. Even if you don't use RSC now, this separation can make it easier to adopt in the future.

### 3.2 Event Handling and Side Effect Patterns

Modern React encourages a clear separation between interactive event handling and passive effect synchronization. Some best practices for events and side effects in React 19:

* **Do side effects in events when possible:** As covered, handle the results of a user action *during* that action’s handler. Update state, call APIs, etc., directly in response to events instead of using an effect to “watch” for something to change. This makes the code more predictable and avoids extra re-renders[\[4\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,in%20the%20corresponding%20event%20handlers).

* **Use stable event handler references wisely:** If passing event handlers down to many child components (e.g., a deeply nested callback prop), consider using useCallback to avoid unnecessary re-renders of those children. However, do not overdo it – only optimize if a real performance issue exists. React’s upcoming optimizations can handle many cases, so focus on correctness first[\[54\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20Write%20straightforward%20JavaScript,compiler%20might%20need%20a%20hint). In React 19, if you find yourself writing a lot of useCallback(\[\]) (empty dependency) just to avoid re-renders, that might hint at using useEffectEvent or a different approach where the child doesn’t need a changing callback prop.

* **Avoid anonymous functions inline if performance-sensitive:** It’s fine to write \<button onClick={() \=\> doSomething(x)}\> most of the time, but if that button is in a list of 1000 items, creating 1000 new lambdas each render could be a concern. In those cases, define the handler outside or use useCallback. With Vite and development mode, you want HMR to be fast, so clean code is key – don’t micro-optimize prematurely, but be aware of hot paths.

* **Clean up effects properly:** Whenever you use useEffect for something like an event listener (e.g., adding a keydown listener on mount), always return a cleanup function to remove it. This sounds basic, but it’s a common oversight that leads to memory leaks or duplicate handlers. React 19 has strict mode running effects twice on mount (in dev) to help catch issues – ensure your effect cleanups can handle being called multiple times (idempotently remove listeners, etc.).

* **Use useEffectEvent for event handlers defined inside effects:** This was discussed earlier – if an effect sets up an event subscription (say a WebSocket or document event) and you need to call component logic from that event, wrap that logic in useEffectEvent. This pattern prevents having to list a bunch of state as dependencies just for that handler and ensures you don’t get stale values[\[5\]](https://react.dev/learn/separating-events-from-effects#:~:text=Here%2C%20,of%20your%20props%20and%20state).

* **Throttle or debounce via React, not manually:** If you need to throttle user events (e.g., a scroll handler), consider using React’s concurrent features rather than manual \_.throttle. For instance, wrap state updates in startTransition to let React drop intermediate updates. Or use useDeferredValue if you have a rapidly updating value. This leverages React’s internal scheduling rather than managing timers yourself.

Overall, treat effects as last-resort escape hatches (for system sync like APIs, logging, etc.), and treat event handlers as the first-class place to handle interactive logic. This aligns with React’s paradigm: render is pure and synchronous, events are where you “step out” to do things like data updates or navigation, and effects are mainly for bridging to the outside world.

### 3.3 State Management, Data Flow, and Derived State

Managing state effectively is a core React skill. Modern best practices emphasize simplicity and single sources of truth:

* **State colocation:** Keep state as *close to where it’s used as possible*. This means if only one component (or a small subtree) needs a piece of state, put that state in that component rather than a higher parent or global store[\[60\]](https://bobaekang.com/blog/component-colocation-composition/#:~:text=React%20bobaekang,This%20mostly%20works). Colocated state is easier to manage and reason about. Lift state up only when necessary to share it – following the principle from “Thinking in React” of identifying the lowest common ancestor that needs the state[\[61\]](https://react.dev/learn/thinking-in-react#:~:text=Thinking%20in%20React%20Identify%20every,them%20all%20in%20the%20hierarchy).

* **Avoid redundant or duplicate state:** Don’t store the same piece of data in two places. This often leads to the out-of-sync bugs. For example, storing a list of items and also storing a filtered list of those items in state is usually unnecessary – store the source and derive the filtered list on the fly (or via memo). The new React docs strongly advise to not duplicate state unless you absolutely need to cache something heavy[\[46\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,your%20props%20or%20state%20change)[\[47\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=function%20Form%28%29%20). If you find yourself writing an effect to keep two state values in sync, that’s a red flag (and an opportunity to refactor).

* **Derived data with selectors or memos:** If deriving data is expensive, use useMemo to avoid recomputation on every render. For instance, const visibleItems \= useMemo(() \=\> filterItems(items, search), \[items, search\]);. This way, the filtering only re-runs when items or search change. But do not use useMemo for every trivial thing – it's mainly for performance optimization. The rule of thumb: derive data in render if you can (it will recompute when needed). Only cache it with useMemo if you have proven performance issues or the computation is heavy.

* **Global state and context:** Global state management libraries (Redux, MobX, Zustand, etc.) can still be useful, but consider context or simpler patterns first for moderate needs. React’s built-in useContext can serve for sharing state globally (like a current user, theme, etc.), but remember that context updates will re-render all components that consume it. For fine-grained performance, external stores (with useSyncExternalStore) or libraries might be better. If using context, a common pattern is to split state and dispatch: e.g., provide state and an update function separately so that components can useContext the state or the dispatch independently. Also, to avoid context tearing with concurrent rendering, stick to the documented patterns (using useSyncExternalStore internally if making a custom context that wraps an external store).

* **Forms and controlled vs uncontrolled:** With new form hooks, you have more choices. You can use the traditional controlled components approach (each input has a state and onChange). Or leverage useFormState to handle multiple fields at once[\[17\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=Now%2C%20with%20the%20new%20,state%20when%20any%20field%20changes). If using actions (like in RSC frameworks or with useActionState), you might even use uncontrolled form fields that get collected on submit (e.g., \<form action={formAction}\> usage). The best practice is to keep forms simple: if a simple useState per field works, that’s fine; for larger forms, consider a library or the new hooks to avoid too much boilerplate. And always handle validation and errors gracefully (client-side validate but also handle server errors via form status or similar).

* **“Source of truth” for data:** If data comes from a server, decide whether the server is the source of truth or your client state is. For example, after a successful mutation, do you trust the client to update its state, or do you re-fetch from the server? With React 19’s actions and libraries like React Query, a common best practice is to optimistically update the UI (using useOptimistic or similar) but also revalidate or update from the authoritative source to keep things consistent. In any case, avoid inconsistent sources of truth (e.g., don’t keep some global singleton data and also React state that mirrors it without syncing – if you have to do it, encapsulate the syncing in one place, e.g., a custom hook or the external store pattern).

* **Use reducers for complex state logic:** If you find yourself managing multiple pieces of state that all update in a coordinated way (like several fields that all reset together, or a state that has sub-properties), using useReducer can make the code cleaner. It centralizes the update logic and can reduce the number of re-renders if done right. It’s also easier to test the reducer logic independently. React 19 doesn’t change useReducer, but with the coming React Compiler optimizations, simple reducers might even get special handling. The rule of thumb: if updates to state A usually accompany updates to state B, consider a reducer.

* **Immutability and avoiding mutations:** Continue to treat state as immutable and avoid directly mutating objects/arrays in state. This is fundamental for enabling React’s optimizations and making state updates predictable. Libraries like Immer can help, or just using spread syntax correctly. With hooks, ensure you replace state (e.g., setX(newX)) rather than mutating in place. This might be basic, but it remains a best practice.

* **Prop drilling vs context:** If you find yourself drilling a prop through many layers just to get it to one deep component, consider context to provide it instead. Prop drilling is fine for 1-2 levels, but beyond that it hurts maintainability. For functions that update state in a parent, you can also use context or a state management solution. However, don’t use context for everything – only for values that truly are needed widely or deeply in the tree.

In summary, **clean data flow** means each piece of data lives in one place and flows down (via props or context) to where it’s used, and updates bubble up through event handlers or context updaters. React 19’s new hooks (like form and action hooks) further reinforce one-directional data flow: e.g., form inputs can update local state, and actions then sync with server.

One more note: consider TypeScript (if you aren’t already using it) to manage your state shapes and props. It’s now considered an industry-standard best practice to use TypeScript with React for maintainability[\[62\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=9,One). With Vite, adding TypeScript is easy and gives you confidence as you refactor state and context.

### 3.4 Performance Optimization Techniques

React 19 brings performance improvements out of the box (like transition API, Suspense improvements, automatic batching). But developers still play a role in writing efficient apps. Key practices include:

* **Code-splitting with Suspense:** Use React.lazy and \<Suspense\> to split your code into chunks that load on demand. Vite’s bundling makes it easy to lazy-load components. For example:

* const SettingsPanel \= React.lazy(() \=\> import('./SettingsPanel'));  
  // ...  
  \<Suspense fallback={\<Spinner /\>}\>  
    {showSettings && \<SettingsPanel /\>}  
  \</Suspense\>

* This ensures that heavy code for rarely-used parts of the UI doesn’t bloat initial load. As a best practice, lazy-load routes or expensive components. React 19 even introduced *preloading* APIs to load code or data ahead of time if needed (check React 19 blog for ReactDOM.preload etc., if relevant).

* **Use transitions for smoother UX:** We covered useTransition – use it when updating large parts of the UI in response to something that doesn't need to update instantly. It can greatly improve perceived performance in complex apps by avoiding locking up the UI. For example, wrapping a navigation state change in startTransition can keep a spinner or previous screen visible until the next screen is ready, rather than freezing the interface.

* **Avoid excessive re-renders:** While React is quite fast, unnecessary renders can still hurt. Tools:

* React.memo for functional components to skip re-render when props haven’t changed. Use this for components that are pure (no side effects in rendering) and often receive the same props. Example: a list item component that re-renders only if its item prop changes.

* useMemo and useCallback to avoid recalculating expensive values or recreating functions that cause downstream renders. However, **don’t memoize everything** – focus on expensive operations or components that you profiled and found problematic. As the best practice article suggests, *“write straightforward JavaScript and let the compiler handle performance optimization. Only reach for manual memoization in rare edge cases.”*[\[54\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20Write%20straightforward%20JavaScript,compiler%20might%20need%20a%20hint). This is a nod to the forthcoming React Compiler (“React Forget”) which aims to auto-memoize where it’s obviously safe. Today, you still use these tools, but you might not need to wrap every single function in useCallback – that can actually harm performance due to memory usage and indirection. Use the React DevTools Profiler to pinpoint bottlenecks.

* **Use useDeferredValue for large lists tied to input:** If you have something like a big table filtering as the user types, useDeferredValue can ensure the table doesn’t thrash on every keypress. We discussed this earlier; it's a simple way to optimize that specific scenario with minimal code.

* **Virtualize long lists:** For extremely large lists (hundreds or thousands of items), consider virtualization (using libraries like react-window or react-virtualized) to render only visible items. This isn’t a React 19-specific tip, but remains a best practice for performance. Vite makes adding such libraries straightforward.

* **Optimize context usage:** If using context for frequently changing values, be mindful: updating context triggers every consumer to re-render. In some cases, splitting context or using an external store can improve performance. E.g., instead of one giant context for your whole app state, use multiple contexts for independent parts so that changing one doesn’t re-render everything.

* **Suspense for data fetching:** If you adopt Suspense for data (with use() or libraries that support it), you get a built-in mechanism for avoiding inconsistent loading states. The UI will “pause” on the Suspense boundary until data is ready, which can simplify logic and avoid intermediate states where some parts are updated and others not. Ensure to provide good fallback UI. Pair Suspense with **Error Boundaries** to catch errors from async code.

* **Concurrent rendering pitfalls:** React 18+ concurrent mode means effects can run double in development (Strict Mode) and components may render and then be thrown away. Make sure your code doesn’t rely on side effects in render or on mount that shouldn’t happen twice. Generally, avoid side effects in the render phase altogether (React rules of hooks enforce this, but it also means avoid doing something like modifying a ref or external variable during render). These practices prevent issues that could degrade performance or cause weird bugs in concurrent scenarios.

* **Profiling and metrics:** Use React Profiler (or useEffect with performance.now() etc.) to measure if an update or render is slow. Optimize based on data, not assumptions. Sometimes what we think is slow isn’t the real issue. Vite's fast refresh and build won’t directly affect runtime performance, but it helps your development loop to iterate on optimizations quickly.

In essence, modern React performance is about using the tools provided (transitions, deferred values, memoization) in targeted ways. And importantly, trust React – it’s pretty optimized. Focus on high-level improvements like code-splitting and reducing work (algorithmically), before micro-optimizing every component.

### 3.5 Component Structure and Co-Location Strategies

How you organize your project and components can greatly affect maintainability. React doesn’t mandate a structure, but some patterns have emerged as best practices:

* **Organize by feature, not type:** Instead of having separate folders for “components,” “hooks,” “utils,” etc., many teams prefer feature-based organization[\[63\]](https://medium.com/@jigsz6391/best-practices-for-structuring-components-in-react-js-e3e29c2a77e3#:~:text=Best%20Practices%20for%20Structuring%20Components,structure%20them%20by%20feature)[\[64\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=As%20projects%20grow%2C%20a%20well,easier%20to%20navigate%20and%20maintain). For example, if you have a feature for user profiles, you might have a features/profile/ folder containing ProfilePage.jsx, ProfileForm.jsx, useProfileData.js, maybe a profileSlice.js if using Redux, etc. This co-locates all logic for that feature. Shared generic components can still live in a common folder (like a components/ for truly reusable pieces). This structure makes it easier to find related code and to scale the codebase without massive monolithic directories.

**Example project structure:**

src/  
  features/  
    authentication/  
      LoginForm.jsx  
      useAuthStatus.js  
      auth.api.js    // maybe API calls  
      auth.slice.js  // state management (if any)  
    products/  
      ProductList.jsx  
      ProductDetail.jsx  
      products.api.js  
    profile/  
      ProfilePage.jsx  
      ProfileForm.jsx  
      useProfileData.js  
  components/  
    Button.jsx  
    Modal.jsx  
  hooks/  
    useViewportSize.js   // generic hooks if any  
  ...

*(The exact structure can vary; the key is grouping by feature.)*

Grouping by feature not only helps co-locate files, but also naturally encourages **component co-location** – small components that are only used by one feature can live alongside it instead of in a global folder.

* **Co-locate test and style files:** It’s often beneficial to keep a component’s CSS (if using CSS modules or stylesheets) and tests next to the component file. For example, ProfilePage.test.jsx and ProfilePage.module.css in the same folder as ProfilePage.jsx. Vite doesn’t constrain this, and it makes it easy to move or delete features as a whole.

* **Prefer composition over complex components:** Build many small, focused components and compose them, instead of one giant component that does everything. For instance, use container/presentational patterns or just break UI into logical pieces. A good rule: if a component exceeds a certain size (say, \~300 lines) or handles disparate concerns, consider splitting it. In React 19, with hooks like useActionState and others, you can keep components smaller by moving logic into custom hooks or into child components. Composition makes reuse easier and testing each piece simpler. As one best practice states: *“Build small, single-responsibility components. Use the children prop to create flexible and reusable layouts and containers.”*[\[65\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L298%20Best%20Practice%3A,and%20reusable%20layouts%20and%20containers).

For example, instead of a monolithic Dashboard component that implements its own layout, make a reusable Card component and compose Dashboard from \<Card\>s for each section. This was shown earlier:

function Card({ title, children }) { ... }   
function UserProfile({ user }) {  
  return (  
    \<Card title={user.name}\>  
      \<p\>Email: {user.email}\</p\>  
      \<p\>Member since: {user.joinDate}\</p\>  
    \</Card\>  
  );  
}

This kind of composition yields cleaner, more maintainable code.

* **Co-locate state logic with the component that uses it:** If you write a custom hook, consider defining it in the same file or same folder as the component that primarily uses it (unless it’s truly generic). For example, if you have a complex piece of state management for a form, you might make useLoginForm inside features/authentication/useLoginForm.js next to LoginForm.jsx. This way, if the feature is removed, everything goes with it. It also signals that the hook is specific to that feature.

* **Use index files for grouping exports:** In each feature folder, you can have an index.js that exports the main components/hooks of that feature. This provides a clean interface if someone needs to import things from that feature. e.g., export \* from './ProfilePage.jsx'; export \* from './useProfileData.js'; so in other parts of app you can do import { ProfilePage } from 'features/profile';. This is optional but can improve ergonomics.

* **File naming conventions:** Follow a consistent convention (e.g., PascalCase for component files like MyComponent.jsx). This helps quickly distinguish components from plain JS modules. Vite will work with any naming, but clarity helps.

* **Avoid deeply nested hierarchies:** While grouping by feature, don’t create too many nested subfolders unless needed. Deep nesting can be cumbersome to navigate. Strike a balance between grouping logically and keeping paths short.

In summary, structure your React 19 project in a way that **keeps related things together and separates unrelated concerns**. A modular structure makes it easier to reason about code and to adopt new patterns (like adding a new hook or replacing a component) without affecting distant parts of the app. It also plays well with source control (fewer merge conflicts if teams work in separate feature folders).

### 3.6 Additional Best Practices and Tips

To round out, here are a few more best practices worth mentioning for modern React development with Vite:

* **Use TypeScript**: As noted earlier, adopting TypeScript from day one is highly recommended for modern React apps[\[62\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=9,One). It catches errors early, provides better documentation via types, and works seamlessly with Vite (which has great TS support). Given React 19’s new APIs (like Actions) have good TypeScript support, you’ll find it easier to use them with proper type checking.

* **Testing**: Write tests that focus on user behavior and outcomes, not implementation details. With hooks and functional components, it’s easier to test because you can test pure functions (like reducers or custom hooks via the React Hooks Testing Library) and you can use React Testing Library to simulate clicks and see rendered output. Aim for tests that reflect real usage: “mount component, simulate user input, assert on output”[\[66\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=10.%20Write%20Meaningful%2C%20User)[\[67\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=test%28,toBeInTheDocument%28%29%3B). This ensures your refactors (like migrating an effect to a hook or a class to a function) don’t break user-facing behavior. Vite’s ecosystem has Vitest which is a great test runner that works similarly to Jest.

* **ESLint and linters**: Use the official React hooks ESLint plugin to catch mistakes with dependencies and rules of hooks. Also consider the eslint-plugin-react-you-might-not-need-an-effect plugin[\[68\]](https://github.com/NickvanDyke/eslint-plugin-react-you-might-not-need-an-effect#:~:text=NickvanDyke%2Feslint,easier%20to%20follow%2C%20faster)[\[69\]](https://react.dev/learn/synchronizing-with-effects#:~:text=Synchronizing%20with%20Effects%20,your%20React%20code%20and), which can warn when you’ve written an effect that likely isn’t necessary. These tools guide you toward best practices by automating the advice from the React docs.

* **Stay updated with React releases**: React is evolving (the React 19.x minor releases might bring more features). Keep an eye on the official blog and docs for changes like compiler improvements, new hooks, etc. For instance, if React 20 introduces a new context API or a new Hook, be ready to incorporate those into your practices.

* **Performance monitoring**: In production, use tools or browser DevTools to monitor performance. Ensure you are splitting chunks (Vite will code-split dynamic imports by default). If using Vite, you can analyze bundle size with plugins to ensure you didn’t accidentally bloat the app.

Using Vite specifically gives you a fast dev environment, so take advantage of features like hot-module replacement to maintain state when editing components (this speeds up UI tweaks immensely). Also, Vite’s production build is optimized – trust it to handle tree-shaking and minification; you just focus on writing clean code.

Finally, always remember the core principle: **React (and its ecosystem) is about declarative UI.** Best practices serve that goal – make your code describe *what* you want to do, not *how* to do it imperatively. The new hooks in React 19, the migration from classes to functions, and patterns like Server Components all push us toward clearer, more declarative code where React handles the heavy lifting.

## 4\. Examples Recap: Old vs New Patterns

To solidify these concepts, let’s recap with a few quick before-and-after comparisons:

* **Derived State (No Effect vs Effect):**

*Before:* Using useEffect to derive state from props:

const \[filtered, setFiltered\] \= useState(\[\]);  
useEffect(() \=\> {  
  setFiltered(items.filter(item \=\> item.active));  
}, \[items\]);

*After:* Derive during render (recompute when items changes):

const filtered \= useMemo(  
  () \=\> items.filter(item \=\> item.active),  
  \[items\]  
);

*(No state or effect needed; using useMemo in case filtering is expensive.)*

* **User Event Handling:**

*Before:* Setting state then effect doing side effect:

const \[copied, setCopied\] \= useState(false);  
useEffect(() \=\> {  
  if (copied) {  
    toast("Copied\!");  
  }  
}, \[copied\]);  
// somewhere in JSX:  
\<button onClick={() \=\> setCopied(true)}\>Copy\</button\>

*After:* Do it in one go:

const handleCopy \= () \=\> {  
  copyToClipboard();  
  toast("Copied\!");  
  // no state necessary unless UI needs to reflect it  
};  
\<button onClick={handleCopy}\>Copy\</button\>

*(Directly performs the action without an intermediate state/effect.)*

* **Async Form Submission (Manual vs useActionState):**

*Before:*

const \[loading, setLoading\] \= useState(false);  
const \[error, setError\] \= useState(null);  
const handleSubmit \= async (data) \=\> {  
  setLoading(true);  
  try {  
    await sendData(data);  
  } catch (err) {  
    setError(err);  
  } finally {  
    setLoading(false);  
  }  
};  
\<form onSubmit={handleSubmit}\>... {loading && "Submitting"}\</form\>

*After (useActionState):*

const \[result, submit\] \= useActionState(async (\_, formData) \=\> {  
  return sendData(Object.fromEntries(formData));   
  // If throws, result.error will be set  
});  
\<form action={submit}\>  
  ... \<button disabled={result.isLoading}\>Submit\</button\>  
  {result.isLoading && "Submitting"}  
  {result.error && \<ErrorMsg error={result.error} /\>}  
\</form\>

*(State flags are handled by the hook; the form uses the action prop to integrate with React’s action system[\[70\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=%3Cform%20action%3D%7BformAction%7D%3E%20%3Cdiv%3E%20%3Cinput%20type%3D,Submit%3C%2Fbutton%3E%20%3C%2Fdiv%3E%20%3C%2Fform)[\[71\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=import%20,error%3A%20null%2C).)*

* **External Store Subscription (Effect vs useSyncExternalStore):**

*Before:*

const \[online, setOnline\] \= useState(navigator.onLine);  
useEffect(() \=\> {  
  const update \= () \=\> setOnline(navigator.onLine);  
  window.addEventListener('online', update);  
  window.addEventListener('offline', update);  
  return () \=\> {  
    window.removeEventListener('online', update);  
    window.removeEventListener('offline', update);  
  };  
}, \[\]);

*After:*

const online \= useSyncExternalStore(subscribe,   
                                    () \=\> navigator.onLine,   
                                    () \=\> true);

(As shown earlier, subscribe encapsulates the addEventListener logic[\[36\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=function%20subscribe%28callback%29%20)[\[35\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=return%20useSyncExternalStore). We get a readable online value directly, no explicit effect needed in the component.)

Each “after” pattern tends to reduce the amount of code and separate concerns more cleanly. Fewer effects mean fewer opportunities for bugs with dependencies or timing. More declarative hooks mean the React runtime can handle details (like tracking isPending, cleaning up stale tasks, etc.) for you.

---

**In conclusion**, React 19 with Vite enables a highly productive and modern development workflow. By embracing the new hooks (useEffectEvent, useActionState, useTransition, useDeferredValue, useSyncExternalStore, etc.) and following best practices (prefer declarative patterns, keep state minimal and localized, utilize Suspense and server-side where possible, and write clean, structured code), you can create React applications that are both robust and easy to maintain. The React team has provided these tools to make our code more expressive – leveraging them will help avoid the “Effect soup” of the past and move into a clearer, more efficient style of React programming[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone)[\[43\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L244%20Best%20Practice%3A,cleaner%2C%20more%20readable%20async%20code).

**Sources:**

* Official React 19 Documentation and Blog[\[72\]](https://react.dev/blog/2024/12/05/react-19#:~:text=What%E2%80%99s%20new%20in%20React%2019)[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone)[\[37\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=)

* React 19 New Hooks (useActionState, useFormStatus, etc.) – freeCodeCamp and Dev.to guides[\[13\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20SubmitButton%28%29%20)[\[14\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=With%20useActionState%3A)[\[73\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=,the%20following%20way)

* React 19 Best Practices – Community articles and posts[\[74\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20For%20static%20content,only%20APIs)[\[75\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L324%20Best%20Practice%3A,reason%20about%20as%20it%20scales)

* React Docs: “You Might Not Need an Effect” – guidelines on when effects are unnecessary[\[46\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,your%20props%20or%20state%20change)[\[4\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,in%20the%20corresponding%20event%20handlers)

* Code examples adapted from React documentation and community tutorials[\[76\]](https://react.dev/learn/separating-events-from-effects#:~:text=%2F%2F%20)[\[51\]](https://react.dev/reference/react/Component#:~:text=In%20the%20above%20example%2C%20the,logic%20as%20a%20single%20Effect)

---

[\[1\]](https://javascript.plainenglish.io/still-using-useeffect-check-out-react-19s-exciting-new-hooks-08a537100ed8?gi=8375feb7d77a#:~:text=Before%20we%20unpack%20the%20new,asynchronous%20tasks%20or%20managing%20forms) [\[2\]](https://javascript.plainenglish.io/still-using-useeffect-check-out-react-19s-exciting-new-hooks-08a537100ed8?gi=8375feb7d77a#:~:text=It%20has%20served%20well%20for,asynchronous%20tasks%20or%20managing%20forms) Still Using UseEffect? Check Out React 19’s Exciting New Hooks | by Oscar Luna | JavaScript in Plain English

[https://javascript.plainenglish.io/still-using-useeffect-check-out-react-19s-exciting-new-hooks-08a537100ed8?gi=8375feb7d77a](https://javascript.plainenglish.io/still-using-useeffect-check-out-react-19s-exciting-new-hooks-08a537100ed8?gi=8375feb7d77a)

[\[3\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=Effects%20are%20an%20escape%20hatch,prone) [\[4\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,in%20the%20corresponding%20event%20handlers) [\[35\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=return%20useSyncExternalStore) [\[36\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=function%20subscribe%28callback%29%20) [\[37\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=) [\[46\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=,your%20props%20or%20state%20change) [\[47\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=function%20Form%28%29%20) [\[49\]](https://react.dev/learn/you-might-not-need-an-effect#:~:text=code%29%2C%20and%20less%20error,what%20should%20go%20into%20state) You Might Not Need an Effect – React

[https://react.dev/learn/you-might-not-need-an-effect](https://react.dev/learn/you-might-not-need-an-effect)

[\[5\]](https://react.dev/learn/separating-events-from-effects#:~:text=Here%2C%20,of%20your%20props%20and%20state) [\[6\]](https://react.dev/learn/separating-events-from-effects#:~:text=Show%20more) [\[7\]](https://react.dev/learn/separating-events-from-effects#:~:text=return%20%28%29%20%3D) [\[9\]](https://react.dev/learn/separating-events-from-effects#:~:text=However%2C%20,a%20dependency%20of%20your%20Effect) [\[10\]](https://react.dev/learn/separating-events-from-effects#:~:text=Declaring%20an%20Effect%20Event) [\[76\]](https://react.dev/learn/separating-events-from-effects#:~:text=%2F%2F%20) Separating Events from Effects – React

[https://react.dev/learn/separating-events-from-effects](https://react.dev/learn/separating-events-from-effects)

[\[8\]](https://github.com/facebook/react/issues/27793#:~:text=,constant%20reference%20to%20a) useEffectEvent should be useCallback without dependency array ...

[https://github.com/facebook/react/issues/27793](https://github.com/facebook/react/issues/27793)

[\[11\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=In%20earlier%20versions%20of%20React%2C,the%20status%20of%20each%20operation) [\[12\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=const%20handleSubmit%20%3D%20async%28%29%20%3D,setError%28%27Submission%20failed%27%29%3B) [\[13\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20SubmitButton%28%29%20) [\[14\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=With%20useActionState%3A) [\[17\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=Now%2C%20with%20the%20new%20,state%20when%20any%20field%20changes) [\[18\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20ProfileForm%28%29%20,name%3A%20%27%27%2C%20email%3A%20%27%27) [\[19\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=function%20LoginForm%28%29%20,%3D%20useFormStatus) [\[20\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=%3Cinput%20name%3D,button) [\[21\]](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6#:~:text=const%20%5BformData%2C%20setFormData%5D%20%3D%20useFormState%28,name%3A%20%27%27%2C%20email%3A%20%27%27) React 19 Hooks Explained: Everything You Need to Know \- DEV Community

[https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6](https://dev.to/vishnusatheesh/react-19-hooks-explained-everything-you-need-to-know-4il6)

[\[15\]](https://react.dev/blog/2024/12/05/react-19#:~:text=Building%20on%20top%20of%20Actions%2C,cases%20for%20Actions%20in%20forms) [\[16\]](https://react.dev/blog/2024/12/05/react-19#:~:text=function%20ChangeName%28,) [\[26\]](https://react.dev/blog/2024/12/05/react-19#:~:text=const%20) [\[27\]](https://react.dev/blog/2024/12/05/react-19#:~:text=The%20async%20transition%20will%20immediately,while%20the%20data%20is%20changing) [\[29\]](https://react.dev/blog/2024/12/05/react-19#:~:text=function%20UpdateName%28) [\[30\]](https://react.dev/blog/2024/12/05/react-19#:~:text=The%20async%20transition%20will%20immediately,while%20the%20data%20is%20changing) [\[72\]](https://react.dev/blog/2024/12/05/react-19#:~:text=What%E2%80%99s%20new%20in%20React%2019) React v19 – React

[https://react.dev/blog/2024/12/05/react-19](https://react.dev/blog/2024/12/05/react-19)

[\[22\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=We%20can%20use%20the%20useTransition,ensure%20a%20good%20user%20experience) [\[23\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=To%20make%20use%20of%20this%2C,inside%20the%20startTransition%20callback%20function) [\[24\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=So%2C%20when%20I%20first%20enter,while%20typing%20into%20the%20textbox) [\[25\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=Now%2C%20if%20I%20enter%20another,while%20typing%20into%20the%20textbox) [\[28\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=So%2C%20when%20I%20first%20enter,while%20typing%20into%20the%20textbox) [\[31\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=useDeferredValue) [\[32\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=This%20hook%20is%20very%20similar,will%20trigger%20a%20transition%20update) [\[33\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=Instead%2C%20we%20can%20use%20the,updates%20to%20take%20place%20faster) [\[34\]](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419#:~:text=a%20received%20prop%20value%20will,trigger%20a%20transition%20update) useTransition and useDeferredValue in React 18 | by Theviyanthan Krishnamohan | Bits and Pieces

[https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419](https://blog.bitsrc.io/usetransition-and-usedeferredvalue-in-react-18-5d8a09f8c3a7?gi=a67ba7729419)

[\[38\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=As%20the%20name%20suggests%2C%20,the%20form%20is%20still%20submitting) [\[39\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=const%20,updateFn) [\[41\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=import%20,boundary) [\[42\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=const%20DataComponent%20%3D%20%28,div%3E%3B) [\[70\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=%3Cform%20action%3D%7BformAction%7D%3E%20%3Cdiv%3E%20%3Cinput%20type%3D,Submit%3C%2Fbutton%3E%20%3C%2Fdiv%3E%20%3C%2Fform) [\[71\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=import%20,error%3A%20null%2C) [\[73\]](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/#:~:text=,the%20following%20way) React 19 – New Hooks Explained with Examples

[https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/](https://www.freecodecamp.org/news/react-19-new-hooks-explained-with-examples/)

[\[40\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L190%20Best%20Practice%3A,instant%20feedback%20on%20their%20interactions) [\[43\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L244%20Best%20Practice%3A,cleaner%2C%20more%20readable%20async%20code) [\[44\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=) [\[54\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20Write%20straightforward%20JavaScript,compiler%20might%20need%20a%20hint) [\[58\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=While%20not%20part%20of%20the,can%20access%20backend%20resources%20directly) [\[59\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20For%20static%20content,only%20APIs) [\[62\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=9,One) [\[64\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=As%20projects%20grow%2C%20a%20well,easier%20to%20navigate%20and%20maintain) [\[65\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L298%20Best%20Practice%3A,and%20reusable%20layouts%20and%20containers) [\[66\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=10.%20Write%20Meaningful%2C%20User) [\[67\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=test%28,toBeInTheDocument%28%29%3B) [\[74\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=Best%20Practice%3A%20For%20static%20content,only%20APIs) [\[75\]](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911#:~:text=match%20at%20L324%20Best%20Practice%3A,reason%20about%20as%20it%20scales) Unlocking the Power of React 19: 10 Best Practices for Modern Development | by Orfeas Voutsaridis | JavaScript in Plain English

[https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911](https://javascript.plainenglish.io/unlocking-the-power-of-react-19-10-best-practices-for-modern-development-fcbc28a348a5?gi=e3d339ca5911)

[\[45\]](https://react.dev/reference/react/Component#:~:text=Note) [\[48\]](https://react.dev/reference/react/Component#:~:text=This%20useEffect%20%20call%20is,example%20you%20can%20play%20with) [\[50\]](https://react.dev/reference/react/Component#:~:text=,using%20them%20in%20new%20code) [\[51\]](https://react.dev/reference/react/Component#:~:text=In%20the%20above%20example%2C%20the,logic%20as%20a%20single%20Effect) [\[52\]](https://react.dev/reference/react/Component#:~:text=useEffect%28%28%29%20%3D) [\[53\]](https://react.dev/reference/react/Component#:~:text=%2F%2F%20) [\[55\]](https://react.dev/reference/react/Component#:~:text=Pitfall) [\[56\]](https://react.dev/reference/react/Component#:~:text=First%2C%20verify%20that%20your%20componentWillUnmount,is%20missing%2C%20add%20it%20first) [\[57\]](https://react.dev/reference/react/Component#:~:text=Next%2C%20verify%20that%20your%20componentDidUpdate,and%20state%2C%20fix%20that%20first) Component – React

[https://react.dev/reference/react/Component](https://react.dev/reference/react/Component)

[\[60\]](https://bobaekang.com/blog/component-colocation-composition/#:~:text=React%20bobaekang,This%20mostly%20works) Component, colocation, composition: A note on the state of React

[https://bobaekang.com/blog/component-colocation-composition/](https://bobaekang.com/blog/component-colocation-composition/)

[\[61\]](https://react.dev/learn/thinking-in-react#:~:text=Thinking%20in%20React%20Identify%20every,them%20all%20in%20the%20hierarchy) Thinking in React

[https://react.dev/learn/thinking-in-react](https://react.dev/learn/thinking-in-react)

[\[63\]](https://medium.com/@jigsz6391/best-practices-for-structuring-components-in-react-js-e3e29c2a77e3#:~:text=Best%20Practices%20for%20Structuring%20Components,structure%20them%20by%20feature) Best Practices for Structuring Components in React.js \- Medium

[https://medium.com/@jigsz6391/best-practices-for-structuring-components-in-react-js-e3e29c2a77e3](https://medium.com/@jigsz6391/best-practices-for-structuring-components-in-react-js-e3e29c2a77e3)

[\[68\]](https://github.com/NickvanDyke/eslint-plugin-react-you-might-not-need-an-effect#:~:text=NickvanDyke%2Feslint,easier%20to%20follow%2C%20faster) NickvanDyke/eslint-plugin-react-you-might-not-need-an-effect \- GitHub

[https://github.com/NickvanDyke/eslint-plugin-react-you-might-not-need-an-effect](https://github.com/NickvanDyke/eslint-plugin-react-you-might-not-need-an-effect)

[\[69\]](https://react.dev/learn/synchronizing-with-effects#:~:text=Synchronizing%20with%20Effects%20,your%20React%20code%20and) Synchronizing with Effects \- React

[https://react.dev/learn/synchronizing-with-effects](https://react.dev/learn/synchronizing-with-effects)