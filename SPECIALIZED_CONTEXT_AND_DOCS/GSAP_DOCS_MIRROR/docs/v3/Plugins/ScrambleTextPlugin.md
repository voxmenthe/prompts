# ScrambleText

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(ScrambleTextPlugin) 
```

#### Minimal usage

```
gsap.to(element, {
  duration: 1, 
  scrambleText: "THIS IS NEW TEXT"
});
```

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/jOjaoYJ?default-tab=result\&theme-id=41164)

## Description[​](#description "Direct link to Description")

Scrambles the text in a DOM element with randomized characters (uppercase by default, but you can define lowercase or a set of custom characters), refreshing new randomized characters at regular intervals while gradually revealing your new text (or the original text) over the course of the tween (left to right by default). Visually it looks like a computer decoding a string of text. Great for rollovers.

## **Config Object**[​](#config-object "Direct link to config-object")

You can simply pass a string of text directly as the `scrambleText` and it'll use the defaults for revealing it, or you can customize the settings by using a generic object with any of the following properties:

* ### Property

  ### Description

  #### text[](#text)

  String - The text that should replace the existing text in the DOM element. If omitted (or if `"{original}"`), the original text will be used.

* #### chars[](#chars)

  String - The characters that should be randomly swapped in to the scrambled portion the text. You can use `"upperCase"`, `"lowerCase"`, `"upperAndLowerCase"`, or a custom string of characters, like `"XO"` or `"TMOWACB"`, or `"jompaWB!^"`, etc. Default: `"upperCase"`.

* #### tweenLength[](#tweenLength)

  Boolean - If the length of the replacement text is different than the original text, the difference will be gradually tweened so that the length doesn’t suddenly jump. For example, if the original text is 50 characters and the replacement text is 100 characters, during the tween the number of characters would gradually move from 50 to 100 instead of jumping immediatley to 100. However, if you’d prefer to have it immediately jump, set `tweenLength` to `false`. Default: `true`.

* #### revealDelay[](#revealDelay)

  Number - If you’d like the reveal (unscrambling) of the new text to be delayed for a certain portion of the tween so that the scrambled text is entirely visible for a while, use revealDelay to define the time you’d like to elapse before the reveal begins. For example, if the tween’s duration is 3 seconds but you’d like the scrambled text to remain entirely visible for first 1 second of the tween, you’d set `revealDelay` to `1`. Default: `0`.

* #### newClass[](#newClass)

  String - If you’d like the new text to have a particular class applied (using a ``tag wrapped around it), use `newClass: "YOUR_CLASS_NAME"`. This makes it easy to create a distinct look for the new text. Default: `null`.

* #### oldClass[](#oldClass)

  String - If you’d like the **old** (original) text to have a particular class applied (using a ``tag wrapped around it), use `oldClass: "YOUR_CLASS_NAME"`. This makes it easy to create a distinct look for the old text. Default: `null`.

* #### speed[](#speed)

  Number - Controls how frequently the scrambled characters are refreshed. The default is `1` but you could slow things down by using `0.2` for example (or any number). Default: `1`.

* #### delimiter[](#delimiter)

  String - By default, each character is replaced one-by-one, but if you’d prefer to have things revealed word-by-word, you could use a delimiter of `" "` (space). Default: `""`.

* #### rightToLeft[](#rightToLeft)

  Boolean - If `true` the text will be revealed from right to left. Default: `false`.

## Usage[​](#usage "Direct link to Usage")

```
//use the defaults
gsap.to(element, {duration: 1, scrambleText: "THIS IS NEW TEXT"});//or customize things:

gsap.to(element, {
  duration: 1, 
  scrambleText: {
    text: "THIS IS NEW TEXT", 
    chars: "XO", 
    revealDelay: 0.5, 
    speed: 0.3, 
    newClass: "myClass"
  }
});
```

## **Demos**[​](#demos "Direct link to demos")

Check out the full collection of [text animation demos](https://codepen.io/collection/ExBwoK) on CodePen.

ScrambleText Demos

Search..

\[x]All

Play Demo videos\[ ]

