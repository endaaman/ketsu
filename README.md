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

This achieves `Acc 0.917 IoU 0.829`.
