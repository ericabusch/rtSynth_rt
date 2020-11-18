# change log

* 1.0.0 bug fix: small error bar bug:

The reason why the error bar of `AC classifier A evidence for B trials` plus `AD classifier A evidence for B trials` is so small is not a bug, but the result of the full rotation.
When I rotate the ABCD in a full cycle, I will get this for the format of ACB and ADB (which is short for what is shown above in code format)

```
bed   chair bench        bed   table bench
bench chair bed          bench table bed  
bed   bench chair        bed   table chair
chair bench bed          chair table bed  
bed   bench table        bed   chair table
table bench bed          table chair bed  
bench bed   chair        bench table chair
chair bed   bench        chair table bench
bench bed   table        bench chair table
table bed   bench        table chair bench
chair bed   table        chair bench table
table bed   chair        table bench chair
```

And take `bed   chair bench`  for example, you can also find the `chair bed   bench` , if you take the mean of these two, that should be a constant.
So the method to solve this “small error bar bug” is to not do full rotation but to do only the essential ones. e.g. AB as bed chair  and as chair bed and as bench table  and as table bench
Doing that would gives us this `ACB ADB` :
```
bed   bench chair        bed   table chair
chair bench bed          chair table bed  
bench bed   table        bench chair table
table bed   bench        table chair bench
```

# action log
* in order to know the effect of include parameter in testMiniclass.py, I will change the testMiniclass_parent.py to do all include parameters for condition5.

