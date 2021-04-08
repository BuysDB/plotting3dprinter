# 2d plotting on a 3d-printer
This python library generates GCODE from SVG paths, allowing a 3d printer to draw images with a pen.

### Attach a pen to the printer
In order to draw images, we need to attach a pen to the printer.
[I've used this model to attach a pen to my Mk3S](https://www.prusaprinters.org/prints/42978-pen-plotter-adapter-for-prusa-mk3s/comments)

### obtain a SVG 
to generate GCODE a vector-based SVG image file is required.
Currently the library only accepts **paths** made out of lines and splines (cubic and quadratic).
Transformed paths and groups are not (yet) supported.

Software to generate SVG images from a bitmap image which worked well for me is [Potrace](http://potrace.sourceforge.net/).

### Attach the pen at the correct height (callibration)
In order to callibrate the height of the pen I've created a small gcode file which just puts the head in the left bottom and moves the head 2mm above the bed. This will be the drawing height (Z). 

Put a paper sheet on the bed, and clamp it to the bed using magnets. Run the CALLIBRATE.GCODE file and attach the pen such that it just touches the paper.

After callibration run PEN_UP.GCODE afterwards to lift the pen back up.

## Convert the SVG to GCODE
In order to just plot the contour lines (strokes) present in the svg use:

`svgto3dprinterplotter example.svg -o example_outline.gcode --stroke`

Send the generated gcode file to your printer and enjoy!

### Filled shapes
By adding the --fill flag, shapes will be filled with horizontal lines.

`svgto3dprinterplotter example.svg -o example_outline.gcode --fill`









