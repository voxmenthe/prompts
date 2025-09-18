# SplitText lines in nested elements

SplitText doesn't natively support splitting nested elements by "lines", but if you really need that we've put together a helper function for it.

```
function nestedLinesSplit(target, vars) {
  var split = SplitText.create(target, vars),
    words = vars.type.indexOf("words") !== -1,
    chars = vars.type.indexOf("chars") !== -1,
    insertAt = function (a, b, i) {
      //insert the elements of array "b" into array "a" at index "i"
      var l = b.length,
        j;
      for (j = 0; j < l; j++) {
        a.splice(i++, 0, b[j]);
      }
      return l;
    },
    children,
    child,
    i;

  if (typeof target === "string") {
    target = document.querySelectorAll(target);
  }
  if (target.length > 1) {
    for (i = 0; i < target.length; i++) {
      split.lines = split.lines.concat(nestedLinesSplit(target[i], vars).lines);
    }
    return split;
  }

  //mark all the words and character  elements as _protected so that we can identify the non-split stuff.
  children = (words ? split.words : []).concat(chars ? split.chars : []);
  for (i = 0; i < children.length; i++) {
    children[i]._protect = true;
  }

  children = split.lines;
  for (i = 0; i < children.length; i++) {
    child = children[i].firstChild;
    //if the first child isn't protected and it's not a text node, we found a nested element that we must bust up into lines.
    if (!child._protect && child.nodeType !== 3) {
      children[i].parentNode.insertBefore(child, children[i]);
      children[i].parentNode.removeChild(children[i]);
      children.splice(i, 1);
      i += insertAt(children, nestedLinesSplit(child, vars).lines, i) - 1;
    }
  }
  return split;
}

//used like
var mySplitText = nestedLinesSplit(assetTexts, { type: "lines" });
```

## Demo[​](#demo "Direct link to Demo")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/d754bcc333a42b5e654e67595951901b?default-tab=result\&theme-id=41164)
