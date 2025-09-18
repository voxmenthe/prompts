# MorphSVGPlugin.convertToPath

### MorphSVGPlugin.convertToPath( shape:\[Element | String], swap:Boolean ) : Array

Converts SVG shapes like `<circle>`, `<rect>`, `<ellipse>`, or `<line>` into `<path>`

#### Parameters

* #### **shape**: \[Element | String]

  An element or a selector string.

* #### **swap**: Boolean

  By default, the resulting \<path> will be swapped directly into the DOM in place of the provided shape element, but you can define `false` for `swap` to prevent that.

### Returns : Array[​](#returns--array "Direct link to Returns : Array")

Returns an Array of all `<path>` elements that were created.

### Details[​](#details "Direct link to Details")

Technically it's only feasible to morph `<path>` elements or `<polyline>`/`<polygon>` elements, but there are plenty of times you will want to morph a `<circle>`, `<rect>`, `<ellipse>`, or `<line>`. This method makes that possible by converting those basic shapes into `<path>` elements. It can be used like so:

```
MorphSVGPlugin.convertToPath("#elementID");
```

You can pass in an element or selector text, so you could also have it convert ALL of those elements with one line:

```
MorphSVGPlugin.convertToPath("circle, rect, ellipse, line, polygon, polyline");
```

This literally swaps in a `<path>` for each one directly in the DOM, and it should look absolutely identical. It'll keep the attributes like "id", etc. intact so that the conversion, you should be able to target the elements just as you would before.

```
//An svg <rect> Like this:
<rect id="endShape" width="100" height="100" fill="red"/>
//becomes
<path id="endShape" fill="red" d="M100,0 v100 h-100 v-100 h100z"></path>
```

Why not automatically do the conversion? Because that's a bit too intrusive and could cause problems. For example, if you had event listeners applied to the original element(s) or references in your own code to those elements. We feel it's best to make sure the developer is aware of and specifically requests this conversion rather than doing it automatically.

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/gagNeR?default-tab=result\&theme-id=41164)

## Notes[​](#notes "Direct link to Notes")

* If you define an `rx` or `ry` attribute on a `<rect>` element, make sure you define **both** (MorphSVGPlugin will default to a value of 0 whereas some browsers default to copying the one that was defined).
