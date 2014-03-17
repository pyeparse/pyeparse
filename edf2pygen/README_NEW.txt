HOWTO: Make wrapper on Linux

1. Install ctypesgen.

2. Install EyeLink software.

3. Run "ctypesgen.py -a -l edfapi -o ../pyeparse/edf/_edf2py.py /usr/include/edf.h"

*Note: For Linux / OSX gcc is fine but details TBD.

4. Run the following to generate the edf2py.py file:

