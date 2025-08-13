# SplitText

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(SplitText) 
```

#### Minimal usage

```
// split elements with the class "split" into words and characters
let split = SplitText.create(".split", { type: "words, chars" });

// now animate the characters in a staggered fashion
gsap.from(split.chars, {
  duration: 1, 
  y: 100,       // animate from 100px below
  autoAlpha: 0, // fade in from opacity: 0 and visibility: hidden
  stagger: 0.05 // 0.05 seconds between each
});
```

Or use the new `onSplit()` syntax available in v3.13.0+:

```
SplitText.create(".split", {
  type: "lines, words",
  mask: "lines",
  autoSplit: true,
  onSplit(self) {
    return gsap.from(self.words, {
      duration: 1, 
      y: 100, 
      autoAlpha: 0, 
      stagger: 0.05
    });
  }
});
```

SplitText is a small JavaScript library that splits an HTML element's text into individual characters, words, and/or lines (each in its own, newly-created element), allowing you to create gorgeous staggered animations. It's highly configurable and smarter than other text splitting tools thanks to features like automatic screen reader accessibility, masking for reveal effects, responsive re-splitting, and much more.

Detailed Walkthrough - Major rewrite in `v3.13.0` - half the size, 14 new features!

[YouTube video player](https://www.youtube.com/embed/L1afzNAhI40?si=yt5g94qIK6vvEyH0)

## Features[​](#features "Direct link to Features")

Feature Highlights

The new v3.13.0+ features are marked below with "\*"

* **Screen reader Accessibility**\* - Adds `aria-label` to the split element(s) and `aria-hidden` to the freshly-created line/word/character elements.
* **Responsive re-splitting**\* - Avoid funky line breaks when resizing or when fonts load with `autoSplit` and `onSplit()`. Offers automatic cleanup and resuming of animations too!
* **Slice right through nested elements**\* - Elements like `<span>`, `<strong>`, and `<a>` that span multiple lines are handled effortlessly with `deepSlice` so they don't stretch lines vertically.
* **Masking**\* - Wrap characters, words or lines with an extra clipping element for easy mask/reveal effects.
* **Integrates seamlessly** with GSAP's [`context()`](/docs/v3/GSAP/gsap.context\(\).md), [`matchMedia()`](/docs/v3/GSAP/gsap.matchMedia\(\).md) and [`useGSAP()`](/resources/React.md)
* **Flexible targeting** - Apply your own class names to characters, words, or lines. Append `"++"` to auto-increment them (e.g. `word1`, `word2`, etc.). Enable `propIndex`\* to apply CSS variables like `--word: 3`.
* **Ignore certain elements**\* - Perhaps you'd like to leave `<sup>` elements unsplit, for example.
* **Supports emojis & more** - SplitText does an excellent job with foreign characters too.
* **Revert anytime** - Restore the element's original `innerHTML` anytime with `revert()`
* **Handle complex edge cases** with custom `RegExp`\* or `prepareText()`\*

## Splitting[​](#splitting "Direct link to Splitting")

### Basic Usage[​](#basic-usage "Direct link to Basic Usage")

Feed `SplitText.create()` the element(s) you'd like to split and it'll return a SplitText instance with `chars`, `words`, and `lines` properties where you can access the resulting elements.

```
// the target can be selector text, an element, or an Array of elements
let split = SplitText.create(".headline");

// Array of characters
split.chars

// Array of words
split.words

// Array of lines
split.lines
```

### Configuration[​](#configuration "Direct link to Configuration")

By default, SplitText will split by `type: "lines, words, chars"` (meaning lines, words, **and** characters) but to maximize performance you should really only split what you need. Use the configuration object to control exactly which components are split apart, or to adjust accessibility settings, or apply your own classes or even apply masking effects.

```
let split = SplitText.create(".split", {
  type: "words, lines", // only split into words and lines (not characters)
  mask: "lines", // adds extra wrapper element around lines with overflow: clip (v3.13.0+)
  linesClass: "line++", // adds "line" class to each line element, plus an incremented one too ("line1", "line2", "line3", etc.)

  // there are many other options - see below for a complete list
});
```

## **Config Object**[​](#config-object "Direct link to config-object")

* ### Property

  ### Description

  #### aria\*[](#aria*)

  "auto" | "hidden" | "none" - SplitText can automatically add `aria` attributes to the split element(s) as well as the line/word/character elements to improve accessibility. The options are:

  * `"auto"` (the default) - adds an [aria label](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-label) to the split element(s), populated by its `textContent`, and also adds [aria hidden](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-hidden) to the line/word/character elements inside the split. This ensures that the text is accessible to the majority of screen readers. **This approach will not honor the semantics or functionality of nested elements.** If you need to ensure that links inside your text content are visible to screen readers, we recommend enabling `aria: "hidden"` and creating a duplicate screen reader-only copy of your text.
  * `"hidden"`: adds [aria hidden](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-hidden) to the split element and all of the line/word/character elements inside the split.
  * `"none"` - does not add any `aria` attributes to the split element or the line/word/character elements inside the split.

  Default: `"auto"`

* #### autoSplit\*[](#autoSplit*)

  Boolean - Helps avoid odd line breaks due to text reflow after the fonts finish loading or if the element's width changes. If `true`, SplitText will revert and re-split whenever the fonts finish loading or when **both** of the following conditions apply:

  1. The width of the split element(s) changes
  2. `"lines"` are split.

  SplitText will even `console.warn()` you if you try splitting before the fonts finish loading and you didn't set `autoSplit: true`

  #### Caution

  When using `autoSplit: true`, make sure to create any animations in an `onSplit()` callback so that the freshly-split line/word/character elements are the ones being animated. If you `return` the animation in the `onSplit()`, SplitText will automatically clean up and synchronize the animation on each re-split.

  ```
  SplitText.create(".split", {
    type: "lines",
    autoSplit: true,
    onSplit: (self) => {
      return gsap.from(self.lines, {
        y: 100,
        opacity: 0,
        stagger: 0.05
      });
    }
  });
  ```

  Default: `false`

* #### charsClass[](#charsClass)

  String - A CSS class applied to each character's `<div>`, making it easy to select. If you add `"++"` to the end of the class name, SplitText will also add a second class of that name but with an incremented number appended, starting at 1. For example, if `charsClass` is `"char++"`, the the first character would have `class="char char1"`, the next would have `class="char char2"`, then `class="char char3"`, etc. Default: `undefined`.

* #### deepSlice\*[](#deepSlice*)

  Boolean - If a nested element like `<strong>` wraps onto multiple lines, SplitText subdivides it accordingly so that it doesn't expand the line vertically. So technically one nested element could be split up into multiple elements. This is only effective for splitting `lines`. Default: `true`.

* #### ignore\*[](#ignore*)

  String | Element - Descendant elements to ignore when splitting (you may use selector text like `".split"` or an Array of elements). They will still exist - they simply won't be split [Demo here](https://codepen.io/GreenSock/pen/JojaebV) Default: `undefined`

* #### linesClass[](#linesClass)

  String - A CSS class applied to each line's `<div>`, making it easy to select. If you add `"++"` to the end of the class name, SplitText will also add a second class of that name but with an incremented number appended, starting at 1. For example, if `linesClass` is `"line++"`, the the first line would have `class="line line1"`, the next would have `class="line line2"`, then `class="line line3"`, etc. Default: `undefined`.

* #### mask\*[](#mask*)

  "lines" | "words" | "chars" - wraps every line or word or character in an *extra* element with `visibility: clip` for much simpler reveal effects. Access them in a "masks" Array on the SplitText instance. If you set a class name for the lines/words/chars, it'll append `"-mask"` for easy selecting. You cannot mask multiple types, so this value should be either "lines" or "words" or "chars" but not a combination. Default: `undefined`

* #### onRevert\*[](#onRevert*)

  Function - A function that gets called whenever the SplitText instance reverts

* #### onSplit\*[](#onSplit*)

  Function - A function that gets called whenever the SplitText instance finishes splitting, including when `autoSplit: true` causes it to re-split, like when the fonts finish loading or when the width of the split element(s) changes (which often makes lines reflow). If you return a GSAP animation (tween or timeline), it will automatically save its `totalTime()` and `revert()` it when the SplitText reverts, and set the new animation's `totalTime()` that's returned in the `onSplit`, making it appear relatively seamless!

* #### prepareText\*[](#prepareText*)

  Function - A function that gets called for each block of text as the split occurs, allowing you to modify each chunk of text right before SplitText runs its splitting logic. For example, you might want to insert some special characters marking where word breaks should occur. The `prepareText()` function receives the raw text as the first argument, and the parent element as the second argument. You should **return** the modified text. This can be useful for non-Latin languages like Chinese, where there are no spaces between words. [Demo here](https://codepen.io/GreenSock/pen/VYYvwoq/f30d0213097fe1c8c5a0a09215a5568f)

* #### propIndex\*[](#propIndex*)

  Boolean - adds a CSS variable to each split element with its index, like `--word: 1`, `--word: 2`, etc. It works for all types (line, word, and char). Default: `false`

* #### reduceWhiteSpace[](#reduceWhiteSpace)

  Boolean - Collapses consecutivewhite space characters into one, as most browsers typically do. Set to `false` if you prefer to maintain multiple consecutive white space characters. Since **v3.13.0** reduceWhiteSpace will honor extra spaces and automatically insert `<br>` tags for line breaks which is useful for `<pre>` content. Default: `true`

* #### smartWrap\*[](#smartWrap*)

  Boolean - If you split by `"chars"` only, you can end up with odd breaks at the very end of lines when characters in the middle of a word flow onto the next line, untethered by natural word-grouping. `smartWrap: true` will wrap words in a `<span>` that has `white-space: nowrap` to keep them grouped (only when you're not splitting by words or lines). This will be ignored if you're splitting by `"words"` or `"lines"`, as it's unnecessary. Default: `false`

* #### tag[](#tag)

  String - By default, SplitText wraps things in `<div>` elements, but you can define any tag like `tag: "span"`. Note that browsers won't render transforms like rotation, scale, skew, etc. on inline elements.

* #### type[](#type)

  String - A comma-delimited list of the split type(s) which can be any combination of the following: `chars`, `words`, or `lines`. This indicates the type of components you’d like split apart into distinct elements. For example, to split apart the characters and words (not lines), you’d use `type: "chars,words"` or to only split apart lines, you’d do `type: "lines"`. In order to avoid odd line breaks, it is best to not split by chars alone (always include words or lines too if you're splitting by characters) or just set `smartWrap: true`. Note: spaces are not considered characters. Default: `"chars,words,lines"`.

* #### wordDelimiter[](#wordDelimiter)

  RegExp | "string" | Object - Normally, words are split at every space character. The `wordDelimiter` property allows you to specify your own custom delimiter for words. For example, if you want to split a hashtag like **#IReallyLoveGSAP** into words, you could insert a zero-width word joiner character (`&#8205;`) between each word like: `#&#8205;I&#8205;Really&#8205;Love&#8205;GSAP` and then set `wordDelimiter: String.fromCharCode(8205)` in the SplitText config object. Since **v3.13.0**, you can specify where to split using a RegExp and also what text to swap in at those spots for ultimate flexibility like

  ```
  wordDelimiter: {delimiter: yourRegExp, replaceWith: "yourReplacement"}
  ```

  Default: `" "` (space)

* #### wordsClass[](#wordsClass)

  String - A CSS class applied to each word's `<div>`, making it easy to select. If you add `"++"` to the end of the class name, SplitText will also add a second class of that name but with an incremented number appended, starting at 1. For example, if `wordsClass` is `"word++"`, the the first word would have `class="word word1"`, the next would have `class="word word2"`, then `class="word word3"`, etc. Default: `undefined`.

## Animating[​](#animating "Direct link to Animating")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/xxmaNYj?default-tab=result\&theme-id=41164)

Once your text is split, you can animate each line, word, or character using GSAP:

```
// split all elements with the class "split" into words and characters
let split = SplitText.create(".split", { type: "words, chars" });

// now animate the characters in a staggered fashion
gsap.from(split.chars, {
  duration: 1, 
  y: 100,         // animate from 100px below
  autoAlpha: 0,   // fade in from opacity: 0 and visibility: hidden
  stagger: 0.05,  // 0.05 seconds between each
});
```

Or use the new `onSplit()` syntax available in v3.13.0+ to do the exact same thing - the main benefit is that the code inside `onSplit()` will execute anytime the SplitText instance **re-splits** in the future (like if you set `autoSplit: true` or if you manually call `split()`):

```
SplitText.create(".split", {
  type: "words, chars",
  onSplit(self) { // runs every time it splits
    gsap.from(self.chars, {
      duration: 1, 
      y: 100, 
      autoAlpha: 0, 
      stagger: 0.05
    });
  }
});
```

### Responsive Line Splitting\*[​](#responsive-line-splitting "Direct link to Responsive Line Splitting*")

If *only* words and/or characters are split, they reflow naturally when the container resizes but if you split by **lines**, each line element encloses around a specific set of words/characters. If the container then resizes narrower *or* if the font loads after the split, for example, the text may reflow causing some of the words to belong in *different* lines (the last word in a line may shift down to the next). The only way to avoid strange line breaks is to re-split (restore the original `innerHTML` and have SplitText run its splitting logic again) so that the line elements enclose the proper words.

Don't worry! SplitText's `autoSplit` saves the day! 🥳 When enabled, it will revert and re-split when fonts finish loading or when **both** of the following conditions apply:

* The width of the split element(s) changes
* "lines" are split.

With `autoSplit` enabled, you should **always** create your animations in the `onSplit()` callback so that if it re-splits later, the resulting animations affect the freshly-created line/word/character elements instead of the ones from the previous split. If you **return** your [`tween`](/docs/v3/Plugins/SplitText/docs/v3/GSAP/Tween/) or [`timeline`](/docs/v3/Plugins/SplitText/docs/v3/GSAP/Timeline/) inside the `onSplit()` callback, your old animation will be safely `reverted()` before the new one is created and SplitText will automatically save the previous animation's `totalTime()` before reverting it, and apply it to the new one so that everything appears relatively seamless! The SplitText instance is passed to the `onSplit()` (below, we call it `self`) so you can access its properties:

```
// whenever you use autoSplit: true, ALWAYS create your animations in the onSplit()
SplitText.create(".split", {
    type: "words,lines",
    autoSplit: true,
    onSplit(self) {
      return gsap.from(self.lines, { // a returned animation gets cleaned up and time-synced on each onSplit() call
        yPercent: 20,
        opacity: 0,
        stagger: 1,
        duration: 3,
        onComplete: () => self.revert() // revert the element to its original (unsplit) state
      });
    }
  });
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/2f1edfd9d9462aa26150669eb528fb5f?default-tab=result\&theme-id=41164)

### Masking\*[​](#masking "Direct link to Masking*")

Masking wraps each line, word or character in an *extra* element with `visibility: clip` for fun reveal effects.

```
SplitText.create(".split", {
    type: "words,lines",
    mask: "words", // <-- this can be "lines" or "words" or "chars"
});
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/a28f78aa1b47e6fa3ad56684564fdf81?default-tab=result\&theme-id=41164)

## Screen Reader Accessibility[​](#screen-reader-accessibility "Direct link to Screen Reader Accessibility")

People who are blind or partially-sighted might use a screen reader which analyzes the content of a site and converts it into speech to help them navigate a website. A screen reader would see the following heading tag and read it out loud.

```
<h1>Heading</h1>
```

Most text splitting libraries simply divide the text into divs which screen readers verbalize **painfully** slowly, letter by letter...

```
<h1>
  <div>H</div>
  <div>e</div>
  <div>a</div>
  <div>d</div>
  <div>i</div>
  <div>n</div>
  <div>g</div>
</h1>
```

### Built-in Aria\*[​](#built-in-aria "Direct link to Built-in Aria*")

To get around this issue, SplitText adds an [`aria-label`](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-label) to the parent element and then hides the child elements with [`aria-hidden`](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-hidden). This ensures that when visually impaired people navigate your site, screen readers will read the `aria-label` instead of the contents of the split elements. 🥳 This approach works for the majority of use-cases and is enabled by default.

```
<h2 aria-label="My Accessible Heading">
  <div aria-hidden="true">My</div>
  <div aria-hidden="true">Accessible</div>
  <div aria-hidden="true">Heading</div>
</h2>
```

### Alternate Strategy for Maximizing Nested Element Accessibility[​](#alternate-strategy-for-maximizing-nested-element-accessibility "Direct link to Alternate Strategy for Maximizing Nested Element Accessibility")

SplitText's built in `aria: "auto"` solution is ideal for most common scenarios, but it won't surface the functionality and meaning of nested elements (like links) to screen readers. If you have complex nested text, you can use the duplication approach described below. Exercise restraint here as duplicating lots of DOM elements can lead to performance lags.

Treat text splitting with care and ensure you test thoroughly!

In the example below, the link may not be recognized as such by some screen readers:

```
<h2 aria-label="This link isn't accessible">
  <div aria-hidden="true">This</div>
  <div aria-hidden="true"><a href="#">link</a></div>
  <div aria-hidden="true">isn't</div>
  <div aria-hidden="true">accessible</div>
</h2>
```

If you need to preserve the semantics and functionality of nested elements - like links, `<strong>` tags or `<em>` tags - we recommend disabling the default aria settings for the SplitText with `aria: "none"`\*, and creating a [screen reader-only](https://css-tricks.com/inclusively-hidden/) duplicate of your element instead. This way, sighted users will see the animated text, while visually impaired people will get the screenreader-only content announced to them.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/12697152a77fa81cefd0e6bb3d87c2da?default-tab=result\&theme-id=41164)

## Reverting[​](#reverting "Direct link to Reverting")

Performance-wise, it can be expensive for browsers to render a lot of nodes/elements, so it's often a good idea to `revert()` your split elements to their original state when you're done animating them. Simply call `revert()` on the SplitText instance to restore the original `innerHTML`:

```
let split = SplitText.create(".split", {type: "words"});
gsap.from(split.words, {
  x: "random(-100, 100)",
  y: "random(-100, 100)",
  stagger: 0.1,
  onComplete: () => split.revert() // <-- restores original innerHTML
})
```

## Tips & Limitations[​](#tips--limitations "Direct link to Tips & Limitations")

Tips & Limitations

* **Characters shift slightly when splitting?** - Some browsers apply kerning between certain characters which is lost when each character is put into its own element, thus the spacing shifts slightly. You can typically eliminate that shift by disabling the kerning with this CSS:

  ```
  font-kerning: none; 
  text-rendering: optimizeSpeed;
  ```

* **Custom Fonts** - If you split before your web fonts are ready, the layout may shift or misalign. To avoid this, either:

  * Wait for the fonts to load before splitting by placing your code inside `document.fonts.ready.then(() => {...your code here...})`, or
  * Set `autoSplit: true` to have SplitText re-split once fonts finish loading. Don't forget to put your animation code inside the `onSplit()` callback!

* **Only split what you need** - Splitting thousands of elements can be expensive. If you’re only animating words or lines, skip splitting characters for better performance.

* **SEO** - If you split your main `<h1/>` element, ensure that your page has the appropriate title and description meta tags and your SplitText has `aria: "auto"` (default) enabled. Without these your split heading may appear in google search results in it's composite parts.

* **Avoid text-wrap: balance** - it interferes with clean text splitting.

* **SVG** - SplitText is not designed to work with SVG `<text>` nodes.

* **Standalone plugin** - SplitText is one of the only GSAP plugins that *can* be used **without** loading GSAP's core.

## **Demos**[​](#demos "Direct link to demos")

Check out the full collection of [text animation demos](https://codepen.io/collection/ExBwoK) on CodePen.

SplitText Demos

Search..

\[x]All

Play Demo videos\[ ]

