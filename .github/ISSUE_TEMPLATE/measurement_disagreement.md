---
name: Measurement disagreement
about: A number in the README did not reproduce
labels: measurement
---

A number stated in this repository is wrong unless it reproduces. If it did not, that is
the most useful report this project can receive.

## Which number

Quote it, and say where it appears (README, KNOWN_LIMITATIONS, outputs/...).

## The command you ran to reproduce it

```
```

## Your result

```
```

## Environment

| | |
|---|---|
| Model shape (layers / d_model) | |
| Steps | |
| PyTorch version | |
| Hardware | |
| Commit or version | |

## Anything that could explain the difference

Different hardware, different thread counts, background load, a different seed, a
different corpus. If you know the number is hardware-dependent, say so.
