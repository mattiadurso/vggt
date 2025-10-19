How to use:

1. install vggt
```
pip install -e .
```

2. import and instantiate wrapper
```
from  vggt.wrapper import VGGTWrapper

vggt = VGGTWrapper()
```

3. Choose input and reconstruct
```
input = "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_2D/imc/data/phototourism/british_museum/set_100/images"
output = "/home/mattia/Desktop/Repos/wrapper_factory/sparse"

rec = vggt.forward(input, output)
```

4. Optionally properly triangulate points using colmap CLI
```
vggt.triangulate_with_colmap(input, output, output)
```