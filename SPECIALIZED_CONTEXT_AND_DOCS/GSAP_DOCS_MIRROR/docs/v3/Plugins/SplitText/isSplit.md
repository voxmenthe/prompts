# isSplit

### isSplit : Boolean

Whether the text is currently split.

### Details[​](#details "Direct link to Details")

A boolean that indicates whether the element is currently split. When `true`, the SplitText instance has performed a split and (depending on which types you split by) the `.chars`, `.words`, `.lines`, or `.masks` arrays will be populated. Use `.revert()` to undo the split and reset `isSplit` to `false`.
