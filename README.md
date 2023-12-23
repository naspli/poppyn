# Poppyn
### Binary Population Data Generation and Plotting

Heatmaps of population have a lot of data, and look awesome. 
But they are hard to parse.
For example, no matter how good your eye is, and how good the legend in the heatmap is,
you can't intuitively tell that two orange pixels are equal to one red pixel.

Poppyn processes population data into a binary form.
Either the area has population >N, or it doesn't.

But it isn't quite that simple. 
This would only show what areas of the world are densely populated.
Poppyn 'flattens' the data, so densely populated areas will spill over,
and sparsely populated areas will group together.
This loses some local geographical accuracy,
but helps massively for simple visual comparison of population.

The implementation of this flattening is in function `select_and_flatten_largest_points`.
To run Poppyn yourself, see `run.py`, or add this directory to your `PYTHONPATH` and 
simply `import poppyn`.

TODO: link to blog post with examples.