# ketsu

## Init

```
$ uv sync
```

and download `data/` dir to the project root.


## Train

```
$ uv run task main train --arch unet16n -B 5
```

This achieves `Acc 0.930 IoU 0.841` [ref](https://github.com/endaaman/ketsu/tree/af3eefaa770f71832b7e0a57b9bf9b103fa075af).
