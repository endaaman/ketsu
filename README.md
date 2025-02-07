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

This achieves `Acc 0.917 IoU 0.829` [ref](https://github.com/endaaman/ketsu/tree/3d85bede17f11f237f14ff3e0e97c3ca05b8556d).
