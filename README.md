# battleship-rl

Project made for Probabilistic Methods for Machine Learning course at TCS
2022/23

# usage

## Dependencies

- `numpy`
- `matplotlib`
- `tensorflow`
- `pygame` (only for visualisation)

## Visualisation

### running

To run visualisation simply run

```
python3 main.py
```

By default it runs visualisation on `standard` board. To change that go to
`main.py` and comment/uncomment appropriate line at the bottom of the file.

### controls

- `Spacebar` to make one shot
- `R` to restart with new board
- left and right arrows to switch between algorithms (without resetting the
  board)

## Plots

All plotting is done in `plotting.py` (histograms) and `training.py` (loss
histories)
